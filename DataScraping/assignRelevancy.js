// Based on how far back the gsame is we assign a vlaue for relevancy
// newest games are closer to 1, older ones are closer to 0, its scaled linearly across the 10292 games

const fs = require('fs');

fs.readFile('/Users/amanbrar/Documents/GitHub/SoccerResultPredictor/DataSets/AllPremierLeagueGames.json', (err, data) => {
    if(err) throw new Error(err);
    const jsonObj = JSON.parse(data);
    jsonObj.forEach((element, index) => {
        const relevancy = (10291 - index)/10291;
        element.relevancy = relevancy;
    });
    const newJSON = JSON.stringify(jsonObj);
    fs.writeFile('/Users/amanbrar/Documents/GitHub/SoccerResultPredictor/DataSets/AllPLGames.json', newJSON, (err) => {
        if(err) throw new Error(err);
    });
});