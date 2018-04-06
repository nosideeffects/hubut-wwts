import * as assert from "assert";
import * as fs from "fs";
import * as msgpack from "msgpack-lite";

// Random numbers utils
let return_v = false;
let v_val = 0.0;
function gaussRandom() {
    if(return_v) {
        return_v = false;
        return v_val;
    }
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const r = u * u + v * v;
    if(r == 0 || r > 1) return gaussRandom();
    const c = Math.sqrt(-2 * Math.log(r) / r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
}
function randf(a, b) { return Math.random()*(b-a)+a; }
function randi(a, b) { return Math.floor(Math.random()*(b-a)+a); }
function randn(mu, std){ return mu+gaussRandom()*std; }

function sig(x: number) : number {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidOutputToDerivative(output) : number {
    return output * (1 - output);
}

function dot(a: number[], b: number[]) : number {
    return a.map(function(x,i) {
        return a[i] * b[i];
    }).reduce(function(m,n) { return m + n; });
}

function zeros(n): Float64Array {
    return new Float64Array(n);
}

function softmax(m: Matrix) {
    const out = new Matrix(m.n, m.d); // probability volume
    let maxval = -999999;
    for(let i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

    let s = 0.0;
    for(let i=0,n=m.w.length;i<n;i++) {
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
    }

    for(let i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

    // no backward pass here needed
    // since we will use the computed probabilities outside
    // to set gradients directly on m
    return out;
}

function maxi(w) {
    // argmax of array w
    let maxv = w[0];
    let maxix = 0;
    const n = w.length;
    for(let i = 1; i<n; i++) {
        const v = w[i];
        if(v > maxv) {
            maxix = i;
            maxv = v;
        }
    }
    return maxix;
}

function samplei(w) {
    // sample argmax from w, assuming w are
    // probabilities that sum to one
    const r = randf(0, 1);
    let x = 0.0;
    let i = 0;
    while(true) {
        x += w[i];
        if(x > r) { return i; }
        i++;
    }
}

function initLSTM(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    const model = {};
    let hidden_size;
    for(let d=0;d<hidden_sizes.length;d++) { // loop over depths
        const prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
        hidden_size = hidden_sizes[d];

        // gates parameters
        model['Wix'+d] = Matrix.random(hidden_size, prev_size , 0, 0.08);
        model['Wih'+d] = Matrix.random(hidden_size, hidden_size , 0, 0.08);
        model['bi'+d] = new Matrix(hidden_size, 1);
        model['Wfx'+d] = Matrix.random(hidden_size, prev_size , 0, 0.08);
        model['Wfh'+d] = Matrix.random(hidden_size, hidden_size , 0, 0.08);
        model['bf'+d] = new Matrix(hidden_size, 1);
        model['Wox'+d] = Matrix.random(hidden_size, prev_size , 0, 0.08);
        model['Woh'+d] = Matrix.random(hidden_size, hidden_size , 0, 0.08);
        model['bo'+d] = new Matrix(hidden_size, 1);
        // cell write params
        model['Wcx'+d] = Matrix.random(hidden_size, prev_size , 0, 0.08);
        model['Wch'+d] = Matrix.random(hidden_size, hidden_size , 0, 0.08);
        model['bc'+d] = new Matrix(hidden_size, 1);
    }
    // decoder params
    model['Whd'] = Matrix.random(output_size, hidden_size, 0, 0.08);
    model['bd'] = new Matrix(output_size, 1);
    return model;
}

function forwardLSTM(G: Graph, model, hidden_sizes: number[], x, prev){
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    let hidden_prevs;
    let cell_prevs;
    if (typeof prev.h === 'undefined') {
        hidden_prevs = [];
        cell_prevs = [];
        for (let d = 0; d < hidden_sizes.length; d++) {
            hidden_prevs.push(new Matrix(hidden_sizes[d], 1));
            cell_prevs.push(new Matrix(hidden_sizes[d], 1));
        }
    } else {
        hidden_prevs = prev.h;
        cell_prevs = prev.c;
    }

    let hidden = [];
    let cell = [];
    for (let d = 0; d < hidden_sizes.length; d++) {

        const input_vector = d === 0 ? x : hidden[d - 1];
        const hidden_prev = hidden_prevs[d];
        const cell_prev = cell_prevs[d];

        // input gate
        const h0 = G.multiply(model['Wix' + d], input_vector);
        const h1 = G.multiply(model['Wih' + d], hidden_prev);
        const input_gate = G.sigmoid(G.add(G.add(h0, h1), model['bi' + d]));

        // forget gate
        const h2 = G.multiply(model['Wfx' + d], input_vector);
        const h3 = G.multiply(model['Wfh' + d], hidden_prev);
        const forget_gate = G.sigmoid(G.add(G.add(h2, h3), model['bf' + d]));

        // output gate
        const h4 = G.multiply(model['Wox' + d], input_vector);
        const h5 = G.multiply(model['Woh' + d], hidden_prev);
        const output_gate = G.sigmoid(G.add(G.add(h4, h5), model['bo' + d]));

        // write operation on cells
        const h6 = G.multiply(model['Wcx' + d], input_vector);
        const h7 = G.multiply(model['Wch' + d], hidden_prev);
        const cell_write = G.tanh(G.add(G.add(h6, h7), model['bc' + d]));

        // compute new cell activation
        const retain_cell = G.elementMultiply(forget_gate, cell_prev); // what do we keep from cell
        const write_cell = G.elementMultiply(input_gate, cell_write); // what do we write to cell
        const cell_d = G.add(retain_cell, write_cell); // new cell contents

        // compute hidden state as gated, saturated cell activations
        const hidden_d = G.elementMultiply(output_gate, G.tanh(cell_d));

        hidden.push(hidden_d);
        cell.push(cell_d);
    }

    // one decoder to outputs at end
    const output = G.add(G.multiply(model['Whd'], hidden[hidden.length - 1]), model['bd']);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'c':cell, 'o' : output};
}

function fillRandn(m, mu, std) {
    let i = 0;
    const n = m.w.length;
    for(; i<n; i++) { m.w[i] = randn(mu, std); }
}
function fillRand(m, lo, hi) {
    let i = 0;
    const n = m.w.length;
    for(; i<n; i++) { m.w[i] = randf(lo, hi); }
}


class Matrix {
    w: Float64Array;
    dw: Float64Array;

    constructor(public n: number, public d: number, allocate: boolean = true) {
        if (allocate) this.w = zeros(n * d);
        this.dw = zeros(n * d)
    }

    static random(n,d,mu,std) : Matrix {
        const m = new Matrix(n, d);
        //fillRandn(m,mu,std);
        fillRand(m,-std,std); // kind of :P
        return m;
    }

    toJSON() {
        return {
            n: this.n,
            d: this.d,
            w: this.w
        };
    }

    fromJSON({n,d,w}) {
        this.n = n;
        this.d = d;
        this.w = zeros(this.n * this.d);
        this.dw = zeros(this.n * this.d);
        for(let i = 0, n = this.n * this.d; i < n; i++) {
            this.w[i] = w[i]; // copy over weights
        }
    }
}

type BackpropStep = {
    func: Function,
    args: Matrix[]
}

class Graph {
    backprop: BackpropStep[] = [];

    constructor(private needs_backprop: boolean = true) {

    }

    backward() {
        for(let i=this.backprop.length-1;i>=0;i--) {
            const backpropStep = this.backprop[i]; // tick!
            backpropStep.func.apply(null, backpropStep.args);
        }
    }

    rowPluck(m: Matrix, ix) {
        // pluck a row of m with index ix and return it as col vector
        assert(ix >= 0 && ix < m.n);
        const d = m.d;
        const out = new Matrix(d, 1);
        let n = d;
        for(let i = 0; i < n; i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.rowPluckBackprop,
                args: [ix, m, out]
            });
        }
        return out;
    }

    static rowPluckBackprop(ix: number, m: Matrix, out: Matrix) {
        const d = m.d;
        for (let i = 0; i < m.d; i++) {
            m.dw[d * ix + i] += out.dw[i];
        }
    }

    tanh(m: Matrix) {
        // tanh nonlinearity
        const out = new Matrix(m.n, m.d);
        let n = m.w.length;
        for(let i=0;i<n;i++) {
            out.w[i] = Math.tanh(m.w[i]);
        }

        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.tanhBackprop,
                args: [m, out]
            });
        }

        return out;
    }

    static tanhBackprop(m: Matrix, out: Matrix) {
        let n = m.w.length;
        for(let i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            const mwi = out.w[i];
            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
        }
    }

    sigmoid(m: Matrix) {
        // sigmoid nonlinearity
        const out = new Matrix(m.n, m.d);
        const n = m.w.length;
        for(let i=0;i<n;i++) {
            out.w[i] = sig(m.w[i]);
        }

        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.sigmoidBackprop,
                args: [m, out]
            });
        }
        return out;
    }

    static sigmoidBackprop(m: Matrix, out: Matrix) {
        const n = m.w.length;
        for(let i = 0; i < n; i++) {
            const mwi = out.w[i];
            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
        }
    }

    relu(m: Matrix) {
        const out = new Matrix(m.n, m.d);
        const n = m.w.length;
        for(let i=0; i<n; i++) {
            out.w[i] = Math.max(0, m.w[i]); // relu
        }
        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.reluBackprop,
                args: [m, out]
            });
        }
        return out;
    }

    static reluBackprop(m: Matrix, out: Matrix) {
        const n = m.w.length;
        for (let i = 0; i < n; i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
        }
    }

    multiply(m1: Matrix, m2: Matrix) {
        // multiply matrices m1 * m2
        assert(m1.d === m2.n, 'matmul dimensions misaligned');

        const n = m1.n;
        const d = m2.d;
        const out = new Matrix(n, d);

        for(let i=0; i<n; i++) { // loop over rows of m1
            for(let j=0; j<d; j++) { // loop over cols of m2
                let dot = 0.0;
                for(let k=0; k<m1.d; k++) { // dot product loop
                    dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
                }
                out.w[d*i+j] = dot;
            }
        }

        // out.w = new Float64Array(rustNeon.multiply(m1.w.buffer, m2.w.buffer, m1.d));

        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.mulBackprop,
                args: [m1, m2, out]
            });
        }

        return out;
    }

    static mulBackprop(m1: Matrix, m2: Matrix, out: Matrix) {
        // rustNeon.multiplyBackpropogate(
        //     m1.w.buffer,
        //     m2.w.buffer,
        //     m1.dw.buffer,
        //     m2.dw.buffer,
        //     out.dw.buffer,
        //     m1.d
        // );
        const d = m2.d;
        for (let i = 0; i < m1.n; i++) { // loop over rows of m1
            for (let j = 0; j < m2.d; j++) { // loop over cols of m2
                for (let k = 0; k < m1.d; k++) { // dot product loop
                    const b = out.dw[d * i + j];
                    m1.dw[m1.d * i + k] += m2.w[m2.d * k + j] * b;
                    m2.dw[m2.d * k + j] += m1.w[m1.d * i + k] * b;
                }
            }
        }
    }

    add(m1: Matrix, m2: Matrix) {
        assert(m1.w.length === m2.w.length);

        const out = new Matrix(m1.n, m1.d);
        for (let i = 0; i < m1.w.length; i++) {
            out.w[i] = m1.w[i] + m2.w[i];
        }
        // out.w = new Float64Array(rustNeon.add(m1.w.buffer, m2.w.buffer));

        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.addBackprop,
                args: [m1, m2, out]
            });
        }
        return out;
    }

    static addBackprop(m1: Matrix, m2: Matrix, out: Matrix) {
        const n = m1.w.length;
        for (let i = 0; i < n; i++) {
            m1.dw[i] += out.dw[i];
            m2.dw[i] += out.dw[i];
        }
    }

    elementMultiply(m1: Matrix, m2: Matrix) {
        const out = new Matrix(m1.n, m1.d);
        for (let i = 0; i < m1.w.length; i++) {
            out.w[i] = m1.w[i] * m2.w[i];
        }

        // out.w = new Float64Array(rustNeon.elementwiseMultiply(m1.w.buffer, m2.w.buffer));

        if(this.needs_backprop) {
            this.backprop.push({
                func: Graph.eltmulBackprop,
                args: [m1, m2, out]
            });
        }
        return out;
    }

    static eltmulBackprop(m1: Matrix, m2: Matrix, out: Matrix) {
        const n = m1.w.length;
        for (let i = 0; i < n; i++) {
            m1.dw[i] += m2.w[i] * out.dw[i];
            m2.dw[i] += m1.w[i] * out.dw[i];
        }
    }
}

class Solver {
    decay_rate = 0.999;
    smooth_eps = 1e-8;
    step_cache = {};

    step(model, step_size, regc, clipval) {
        // perform parameter update
        const solver_stats = {};
        let num_clipped = 0;
        let num_tot = 0;
        for(let k in model) {
            if(model.hasOwnProperty(k)) {
                const m = model[k]; // mat ref
                if(!(k in this.step_cache)) { this.step_cache[k] = new Matrix(m.n, m.d); }
                const s = this.step_cache[k];
                const n = m.w.length;
                for(let i = 0; i<n; i++) {

                    // rmsprop adaptive learning rate
                    let mdwi = m.dw[i];
                    s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

                    // gradient clip
                    if(mdwi > clipval) {
                        mdwi = clipval;
                        num_clipped++;
                    }
                    if(mdwi < -clipval) {
                        mdwi = -clipval;
                        num_clipped++;
                    }
                    num_tot++;

                    // update (and regularize)
                    m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
                    m.dw[i] = 0; // reset gradients for next iteration
                }
            }
        }
        solver_stats['ratio_clipped'] = num_clipped/num_tot;
        return solver_stats;
    }
}


// prediction params
const sample_softmax_temperature = 1.0; // how peaky model predictions should be
const max_chars_gen = 100; // max length of generated sentences
// various global var inits
let epoch_size = -1;
let input_size = -1;
let output_size = -1;
let letterToIndex = {};
let indexToLetter = {};
let letter_size = 8;
let hidden_sizes = [128, 128];
let vocab = [];
let data_sents = [];
let solver = new Solver(); // should be class because it needs memory for step caches
let pplGraph = new Graph();
let model = {};

// optimization
let regc = 0.000001; // L2 regularization strength
let learning_rate = 0.003; // learning rate
let clipval = 5.0; // clip gradients at this value

function initVocab(sents, count_threshold) {
    // go over all characters and keep track of all unique ones seen
    const txt = sents.join(''); // concat all
    // count up all characters
    const d = {};
    const n = txt.length;
    for (let i = 0; i < n; i++) {
        const txti = txt[i];
        if (txti in d) {
            d[txti] += 1;
        }
        else {
            d[txti] = 1;
        }
    }
    // filter by count threshold and create pointers
    letterToIndex = {};
    indexToLetter = {};
    vocab = [];
    // NOTE: start at one because we will have START and END tokens!
    // that is, START token will be index 0 in model letter vectors
    // and END token will be index 0 in the next character softmax
    let q = 1;
    for (let ch in d) {
        if (d.hasOwnProperty(ch)) {
            if (d[ch] >= count_threshold) {
                // add character to vocab
                letterToIndex[ch] = q;
                indexToLetter[q] = ch;
                vocab.push(ch);
                q++;
            }
        }
    }
    // globals written: indexToLetter, letterToIndex, vocab (list), and:
    input_size = vocab.length + 1;
    output_size = vocab.length + 1;
    epoch_size = sents.length;

    console.log('found ' + vocab.length + ' distinct characters: ' + vocab.join(''));
}

function utilAddToModel(modelto, modelfrom) {
    for (let k in modelfrom) {
        if (modelfrom.hasOwnProperty(k)) {
            // copy over the pointer but change the key to use the append
            modelto[k] = modelfrom[k];
        }
    }
}

function initModel() {
    // letter embedding vectors
    const model = {};
    model['Wil'] = Matrix.random(input_size, letter_size, 0, 0.08);

    const lstm = initLSTM(letter_size, hidden_sizes, output_size);
    utilAddToModel(model, lstm);

    return model;
}

function reinit() {
    // note: reinit writes global vars

    // eval options to set some globals
    solver = new Solver(); // reinit solver
    pplGraph = new Graph();
    tick_iter = 0;
    // process the input, filter out blanks
    const toddText = fs.readFileSync('todd-no-users.txt', 'utf8');
    const data_sents_raw = toddText.split('\n');
    data_sents = [];
    for (let i = 0; i < data_sents_raw.length; i++) {
        const sent = data_sents_raw[i].trim();
        if (sent.length > 0) {
            data_sents.push(sent);
        }
    }
    initVocab(data_sents, 1); // takes count threshold for characters
    model = initModel();
}

function saveModel() {
    const out = {};
    out['hidden_sizes'] = hidden_sizes;
    out['letter_size'] = letter_size;
    const model_out = {};
    for (let k in model) {
        if (model.hasOwnProperty(k)) {
            model_out[k] = model[k].toJSON();
        }
    }
    out['model'] = model_out;
    const solver_out = {};
    solver_out['decay_rate'] = solver.decay_rate;
    solver_out['smooth_eps'] = solver.smooth_eps;
    const step_cache_out = {};
    for (let k in solver.step_cache) {
        if (solver.step_cache.hasOwnProperty(k)) {
            step_cache_out[k] = solver.step_cache[k].toJSON();
        }
    }
    solver_out['step_cache'] = step_cache_out;
    out['solver'] = solver_out;
    out['letterToIndex'] = letterToIndex;
    out['indexToLetter'] = indexToLetter;
    out['vocab'] = vocab;

    fs.writeFileSync('lstm-trained.json', JSON.stringify(out));
    const buffer = msgpack.encode(out);
    fs.writeFileSync('lstm-trained.msp', buffer);
}

function loadModel(j) {
    hidden_sizes = j.hidden_sizes;
    letter_size = j.letter_size;
    model = {};
    for (let k in j.model) {
        if (j.model.hasOwnProperty(k)) {
            const matjson = j.model[k];
            model[k] = new Matrix(1, 1);
            model[k].fromJSON(matjson);
        }
    }
    solver = new Solver(); // have to reinit the solver since model changed
    solver.decay_rate = j.solver.decay_rate;
    solver.smooth_eps = j.solver.smooth_eps;
    solver.step_cache = {};
    for (let k in j.solver.step_cache) {
        if (j.solver.step_cache.hasOwnProperty(k)) {
            const matjson = j.solver.step_cache[k];
            solver.step_cache[k] = new Matrix(1, 1);
            solver.step_cache[k].fromJSON(matjson);
        }
    }
    letterToIndex = j['letterToIndex'];
    indexToLetter = j['indexToLetter'];
    vocab = j['vocab'];
    tick_iter = 0;
}

function attemptToLoadSaved(fileName: string = 'lstm-trained.msp') {
    if (fs.existsSync(fileName)) {
        console.log('Loading model...');
        const fileContents = fs.readFileSync(fileName);
        try {
            const json = msgpack.decode(fileContents);
            loadModel(json);
        } catch (e) {
            console.error('Failed to load pretrained model', e);
        }
    } else {
        console.log('No pretrained model exists.. starting from scratch..');
    }
}

function forwardIndex(G, model, ix, prev) {
    const x = G.rowPluck(model['Wil'], ix);
    // forward prop the sequence learner
    const out_struct = forwardLSTM(G, model, hidden_sizes, x, prev);

    return out_struct;
}

function predictSentence(model, _samplei: boolean = false, temperature = 1.0, maxCharactersGenerated = max_chars_gen) : string {
    const G = new Graph(false);
    let s = '';
    let prev = {};
    while (true) {
        // RNN tick
        let ix = s.length === 0 ? 0 : letterToIndex[s[s.length - 1]];
        const lh = forwardIndex(G, model, ix, prev);
        prev = lh;
        // sample predicted letter
        const logprobs = lh.o;
        if (temperature !== 1.0 && _samplei) {
            // scale log probabilities by temperature and renormalize
            // if temperature is high, logprobs will go towards zero
            // and the softmax outputs will be more diffuse. if temperature is
            // very low, the softmax outputs will be more peaky
            let q = 0;
            const nq = logprobs.w.length;
            for (; q < nq; q++) {
                logprobs.w[q] /= temperature;
            }
        }
        const probs = softmax(logprobs);
        if (_samplei) {
            ix = samplei(probs.w);
        } else {
            ix = maxi(probs.w);
        }

        if (ix === 0) break; // END token predicted, break out
        if (s.length > maxCharactersGenerated) {
            break;
        } // something is wrong
        const letter = indexToLetter[ix];
        s += letter;
    }

    return s;
}

function costfun(model, sentence: string) {
    // takes a model and a sentence and
    // calculates the loss. Also returns the Graph
    // object which can be used to do backprop
    const n = sentence.length;
    const G = new Graph();
    let log2ppl = 0.0;
    let cost = 0.0;
    let prev = {};
    for (let i = -1; i < n; i++) {
        // start and end tokens are zeros
        const ix_source = i === -1 ? 0 : letterToIndex[sentence[i]]; // first step: start with START token
        const ix_target = i === n - 1 ? 0 : letterToIndex[sentence[i + 1]]; // last step: end with END token
        const lh = forwardIndex(G, model, ix_source, prev);
        prev = lh;
        // set gradients into logprobabilities
        const logprobs = lh.o; // interpret output as logprobs
        const probs = softmax(logprobs); // compute the softmax probabilities
        log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
        cost += -Math.log(probs.w[ix_target]);
        // write gradients into log probabilities
        logprobs.dw = probs.w;
        logprobs.dw[ix_target] -= 1
    }
    const ppl = Math.pow(2, log2ppl / (n - 1));
    return {'G': G, 'ppl': ppl, 'cost': cost};
}

function median(values) {
    values.sort(function (a, b) {
        return a - b;
    });
    const half = Math.floor(values.length / 2);
    if (values.length % 2) return values[half];
    else return (values[half - 1] + values[half]) / 2.0;
}

let tick_iter = 0;

function tick() {
    // sample sentence fromd data
    const sentix = randi(0, data_sents.length);
    const sent = data_sents[sentix];
    const t0 = +new Date();  // log start timestamp
    // evaluate cost function on a sentence
    const cost_struct = costfun(model, sent);

    // use built up graph to compute backprop (set .dw fields in mats)
    cost_struct.G.backward();
    // perform param update
    const solver_stats = solver.step(model, learning_rate, regc, clipval);
    //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)
    const t1 = +new Date();
    const tick_time = t1 - t0;
    // evaluate now and then
    tick_iter += 1;

    if (tick_iter % 100 === 0) {
        console.log('\nSamples:');
        // draw samples
        for (let q = 0; q < 5; q++) {
            let pred = predictSentence(model, true, sample_softmax_temperature);
            console.log(pred);
        }
        console.log('\n');

        // draw argmax prediction
        console.log('argmax prediction: ');
        const pred = predictSentence(model, false);
        console.log(pred);
        // keep track of perplexity

        console.log('\nepoch: ' + (tick_iter / epoch_size).toFixed(2));
        console.log('perplexity: ' + cost_struct.ppl.toFixed(2));
        console.log('forw/bwd time per example: ' + tick_time.toFixed(1) + 'ms');

        // console.log('\nTaking snapshot...')
        // const snapshot1 = profiler.takeSnapshot();
        // // Export snapshot to file file
        // console.log('\nSaving snapshot...')
        // snapshot1.export(function(error, result) {
        //     fs.writeFileSync('snapshot1.json', result);
        //     snapshot1.delete();
        //     process.exit(0);
        // });
    }

    if (tick_iter % epoch_size === 0) {
        console.log('Saving model...');
        saveModel();
        console.log('Saved!')
    }
}

function gradCheck() {
    let model = initModel();
    let sent = '^test sentence$';
    let cost_struct = costfun(model, sent);
    cost_struct.G.backward();
    let eps = 0.000001;
    for (let k in model) {
        if (model.hasOwnProperty(k)) {
            let m = model[k]; // mat ref
            for (let i = 0, n = m.w.length; i < n; i++) {

                let oldval = m.w[i];
                m.w[i] = oldval + eps;
                let c0 = costfun(model, sent);
                m.w[i] = oldval - eps;
                let c1 = costfun(model, sent);
                m.w[i] = oldval;
                let gnum = (c0.cost - c1.cost) / (2 * eps);
                let ganal = m.dw[i];
                let relerr = (gnum - ganal) / (Math.abs(gnum) + Math.abs(ganal));
                if (relerr > 1e-1) {
                    console.log(k + ': numeric: ' + gnum + ', analytic: ' + ganal + ', err: ' + relerr);
                }
            }
        }
    }
}

export class LSTM {
    static import(filename: string = 'lstm-trained.msp') {
        if (!fs.existsSync(filename)) {
            throw new Error('File not found: ' + filename);
        }
        attemptToLoadSaved(filename);
    }

    static initialize() {
        reinit();
    }

    static train() {
        setInterval(tick, 0);
    }

    static sample(maxCharacters: number = 300) : string {
        return predictSentence(model, true, 1.0, maxCharacters);
    }
}
