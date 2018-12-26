const fs = require('fs');

fs.readFile('../DataSets/AllPremierLeagueGames.json', (err, data) => {
    if(err) throw new Error(err);
    const jsonObj = JSON.parse(data);
    const teamsObj = new Map();
    let i = 1;
    jsonObj.forEach((element) => {
        if(!teamsObj.has(element['home'])){
            teamsObj.set(element['home'],i)
            i++
        }
        if(!teamsObj.has(element['away'])){
            teamsObj.set(element['away'],i)
            i++
        }
    });
    console.log(teamsObj);
});