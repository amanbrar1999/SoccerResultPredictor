const Promise = require('bluebird');
const cheerio = require('cheerio');
const Nightmare = require('nightmare')
const nightmare = Nightmare({ electronPath: require('electron'), show: true });
const fs = require('fs');

const baseURL = 'https://www.premierleague.com/results';

async function scrapeSinglePage(){
    await Promise.fromCallback((cb) => nightmare
        .wait('.matchList')
        .then(cb));
    let pixels = 2000;
    while(pixels < 60000){
        await Promise.fromCallback((cb) => nightmare.scrollTo(pixels,0).wait(2000).then(cb));
        pixels += 500;
    }
    const page = await Promise.fromCallback((cb) => nightmare
        .evaluate(() => document.body.innerHTML)
        .then((body) => cb(null, body)));
    const $ = cheerio.load(page);
    return $('.matchList').find('li').map((index, elem) => {
        return {
            home: $(elem).attr('data-home'),
            away: $(elem).attr('data-away'),
            goals_home: $(elem).find('.score').text().split('-')[0],
            goals_away: $(elem).find('.score').text().split('-')[1],
        }
    }).toArray();
}

async function scrape(){
    const page = await Promise.fromCallback((cb) => nightmare
        .goto(baseURL)
        .wait('.matchList')
        .scrollTo(100000,0)
        .wait(5000)
        .evaluate(() => document.body.innerHTML)
        .then((body) => cb(null, body)));
    const $ = cheerio.load(page);
    const yearSelector = '#mainContent > div > div.wrapper.col-12.active > section > div:nth-child(3) > ul';
    const seasons = $(yearSelector).find('li').toArray();
    let fullDataTable = [];
    await Promise.each(seasons, async (season) => {
        const index = parseInt(season.attribs['data-option-index']) + 1;
        await Promise.fromCallback((cb) => nightmare.scrollTo(0,0).click(`${yearSelector} > li:nth-child(${index})`).then(cb));
        const newList = await scrapeSinglePage();
          fullDataTable = fullDataTable.concat(newList);
    });
    const jsonData = JSON.stringify(fullDataTable);
    console.log(fullDataTable.length);
    fs.writeFile('AllPremierLeagueGames.json', jsonData, (err) => {
        if(err) throw new Error(err);
    });
    nightmare.end();
}

scrape();