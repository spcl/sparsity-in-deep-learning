TurndownService = require('turndown')
fs = require('fs')
fs.readFile('index.html', 'utf8', function (err,data) {
    if (err) {
        return console.log('ERROR reading file: ' + err);
    }
    let turndownService = new TurndownService();
    let markdown = turndownService.turndown(data);
    fs.writeFile('README.md', markdown, function (err) {
        if (err) return console.log('ERROR writing file: ' + err);
    });
});


