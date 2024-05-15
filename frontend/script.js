function sendToBackend() {
  var text = document.getElementById("input-textarea").value;

  if (text === "") {
    console.log("No text");
    return;
  } else {
    console.log("Sending to backend");
    console.log(text);
  }

  // URL of backend endpoint
  const url = "http://localhost:5000/predict";

  // Data to be sent in the request body
  const data = {
    text: text,
  };

  // Options for the fetch request
  const options = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  };

  const loadingText = document.getElementById("loadingText")
  loadingText.style.display = "block";
  // Make the fetch request
  fetch(url, options)
    .then((response) => response.json())
    .then((data) => {
      // Handle the response from the backend
      console.log("Response from backend:", data);

      loadingText.style.display = "none";

      const explanation = data.explanation;
      const originalText = text;
      const highlightedText = highlightWords(originalText, explanation);

      const predictionTextDiv = document.getElementById("responsePredictionText");
      predictionTextDiv.innerHTML = ""; // Clear existing content
      // Display prediction
      const prediction = data.prediction_fake;
      let predictionText = "";
      const confidence = Math.abs(0.5 - prediction) * 2 * 100; // Convert to percentage
      if (prediction >= 0.5) {
        predictionText = `This news is probably <span style="color: blue;">REAL</span>. Confidence Level: ${confidence.toFixed(
          2
        )}%`;
      } else {
        predictionText = `This news is probably <span style="color: red;">FAKE</span>. Confidence Level: ${confidence.toFixed(
          2
        )}%`;
      }
      predictionTextDiv.innerHTML += `<p>${predictionText}</p>`;

      // Display highlighted text
      const highlightTextDiv = document.getElementById("responseTextHighlights")
      highlightTextDiv.innerHTML = "<strong>Most important words:</strong>";
      highlightTextDiv.innerHTML += `<p>${highlightedText}</p>`;

    })
    .catch((error) => {
      // Handle any errors that occur during the fetch request
      console.error("Error:", error);
    });

}

// Function to highlight words in the text based on their scores
function highlightWords(text, explanation) {
    // Initialize min and max values for positive and negative scores
    let minPositiveScore = Infinity;
    let maxPositiveScore = -Infinity;
    let minNegativeScore = Infinity;
    let maxNegativeScore = -Infinity;

    // Calculate min and max values for positive and negative scores separately
    explanation.forEach(([_, score]) => {
        if (score >= 0) {
            minPositiveScore = Math.min(minPositiveScore, score);
            maxPositiveScore = Math.max(maxPositiveScore, score);
        } else {
            minNegativeScore = Math.min(minNegativeScore, score);
            maxNegativeScore = Math.max(maxNegativeScore, score);
        }
    });

    // Apply opacity normalization separately for positive and negative scores
    explanation.forEach(([word, score]) => {
        let normalizedOpacity;
        let color;

        if (score >= 0) {
            const normalizedScore = (score - minPositiveScore) / (maxPositiveScore - minPositiveScore);
            normalizedOpacity = Math.abs(normalizedScore * 0.8) + 0.2;
            color = `rgba(0, 0, 255, ${normalizedOpacity})`; // Blue with adjusted opacity
        } else {
            const normalizedScore = (score - minNegativeScore) / (maxNegativeScore - minNegativeScore);
            normalizedOpacity = Math.abs(normalizedScore * 0.8) + 0.2;
            color = `rgba(255, 165, 0, ${normalizedOpacity})`; // Orange with adjusted opacity
        }

        // Apply highlight with adjusted opacity for background color
        text = text.replace(
            new RegExp("\\b" + word + "\\b", "gi"),
            `<span style="background-color: ${color};">${word}</span>`
        );
    });

    return text;
}
