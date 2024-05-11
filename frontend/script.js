function detectFakeNews() {
    var inputText = document.getElementById("inputText").value.trim();
    var outputResult = document.getElementById("outputResult");
    var similarList = document.getElementById("similarList");

    // Placeholder for your fake news detection algorithm
    var isFake = fakeNewsDetectionAlgorithm(inputText);

    if (isFake) {
        outputResult.textContent = "fake";
        outputResult.classList.add("red-text");
    } else {
        outputResult.textContent = "real";
        outputResult.classList.remove("red-text");
    }

    // Get similar articles and words dynamically
    var similarArticlesAndWords = getSimilarArticlesAndWords(inputText);
    similarList.innerHTML = ""; // Clear previous content

    similarArticlesAndWords.forEach(function (item) {
        var card = document.createElement("div");
        card.className = "card mb-2";

        var cardBody = document.createElement("div");
        cardBody.className = "card-body";

        var cardText = document.createElement("p");
        cardText.className = "card-text";
        cardText.textContent = item;

        cardBody.appendChild(cardText);
        card.appendChild(cardBody);
        similarList.appendChild(card);
    });

    similarList.classList.add("scrollable"); // Add scrollable class
}

// Placeholder for your fake news detection algorithm
function fakeNewsDetectionAlgorithm(text) {
    // Implement your detection logic here
    // For demonstration purposes, always return false (not fake)
    return false;
}

// Placeholder for getting similar articles and words
function getSimilarArticlesAndWords(text) {
    // Implement your logic to fetch or generate similar articles and words dynamically
    // For demonstration purposes, return some sample data
    return [
        "Breaking News: New Study Reveals Startling Statistics",
        "Exclusive Interview with Renowned Expert on Climate Change",
        "Local Community Takes Action to Combat Homelessness",
        "World Leaders Gather for Summit on Global Security",
        "Celebrity Scandal Rocks Social Media",
        "Important Word 1", "Impactful Word 2", "Innovative Word 3", "Concerning Word 4",
        "Significant Word 5", "Unexpected Word 6", "Critical Word 7", "Compelling Word 8",
        "Provocative Word 9", "Insightful Word 10"
    ];
}

function sendToBackend(text) {
    console.log("Sending to backend")
    // URL of your backend endpoint
    const url = 'http://localhost:5000/predict'; // Replace with your actual backend URL
    
    // Data to be sent in the request body
    const data = {
        text: text
    };

    // Options for the fetch request
    const options = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    };

    // Make the fetch request
    fetch(url, options)
        .then(response => response.json())
        .then(data => {
            // Handle the response from the backend
            console.log('Response from backend:', data);
            // Process the response data here
        })
        .catch(error => {
            // Handle any errors that occur during the fetch request
            console.error('Error:', error);
        });
}

text = `
At least 60 people have died and more than 100 have been injured after flash flooding in northern Afghanistan, according to Taliban officials.

Dozens of people remain missing after heavy rainfall hit five districts in Baghlan province, with warnings the death toll could rise ahead of a further two storms forecast to spread across the region on Friday night.

Pictures on social media showed torrents of water sweeping through houses in several villages, leaving a trail of destruction in its wake.

The country has been hit by unusually heavy rainfall over the last few weeks, with floods killing more than 100 people since mid-April.

Abdul Mateen Qani, a spokesman for Afghanistan's interior ministry, told the BBC those who had died came from the Borka district in Baghlan province.

More than 200 people have been trapped inside their homes there.

The official earlier told Reuters news agency that helicopters had been sent to Baghlan - located directly north of the capital, Kabul - but "the operation may not be successful" due to a shortage of night vision lights.

Meanwhile, local official Hedayatullah Hamdard told AFP news agency emergency personnel including the army were "searching for any possible victims under the mud and rubble".

Tents, blankets and food were provided to some families who had lost their homes, the official added.

The main road connecting Kabul to northern Afghanistan is closed.

It comes after flooding last month in the west of the country killed dozens of people, leaving thousands requiring humanitarian aid.

About 2,000 homes, three mosques, and four schools were also damaged.

Flash flooding happens when rain falls so heavily that normal drainage cannot cope.

Experts say a relatively dry winter has made it more difficult for the soil to absorb rainfall.

Torrential rain and flooding kill people every year in Afghanistan, where badly built houses in isolated rural areas are particularly vulnerable.

Afghanistan is among the globe's most at risk nations from the effects of climate change, according to experts.

The nation is one of the poorest in the world, having been ravaged by decades of war which culminated in the withdrawal of a US-led coalition and the Taliban retaking control in 2021.

Many factors contribute to flooding, but a warming atmosphere caused by climate change makes extreme rainfall more likely.

The world has already warmed by about 1.1C since the industrial era began and temperatures will keep rising unless governments around the world make steep cuts to emissions.
`
sendToBackend(text)
