// Description
//   What would Todd say? Ask Toddbot!
//
// Configuration:
//   LIST_OF_ENV_VARS_TO_SET
//
// Commands:
//   hubot hello - <what the respond trigger does>
//   orly - <what the hear trigger does>
//
// Notes:
//   <optional notes required for the script>
//
// Author:
//   Jacob Biggs <biggs.jacob@gmail.com>

import {LSTM} from './LSTM';
import * as path from 'path';

module.exports = function (robot) {
  LSTM.import(path.resolve(__dirname, 'lstm-trained.msp'));

  function getUsers() {
    const usernames = [];

    if(robot.brain.data.users && usernames.length === 0){
      for (let user in robot.brain.data.users){
        usernames.push(robot.brain.data.users[user].name);
      }
    }

    return usernames;
  }
  // root
  robot.hear(/(wwtb?s|what would todd(bot)? say)\??/i, (response) => {
    const users = getUsers();
    let sample;
    do {
      sample = LSTM.sample().trim();
    } while (sample.length < 1);

    while (sample.indexOf('USER') > -1) {
      sample = sample.replace('USER', '@' + response.random(users));
    }

    response.send(`"${sample}" - Toddbot`);
  });
};
