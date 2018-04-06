'use strict';

const Helper = require('hubot-test-helper');
const chai = require('chai');
const expect = chai.expect;
const helper = new Helper('../bin/wwts.js');

describe('wwts', function () {
  beforeEach(function () {
    return this.room = helper.createRoom();
  });
  afterEach(function () {
    return this.room.destroy();
  });
  return it('responds to hello', function () {
    return this.room.user.say('alice', 'wwts?').then((function (_this) {
      return function () {
        return expect(_this.room.messages).to.eql([['alice', '@hubot hello'], ['hubot', '@alice hello!']]);
      };
    })(this));
  });
});
