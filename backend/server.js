const express = require("express");
const cors = require("cors");
const { exec } = require("child_process");
const path = require("path");
const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
    res.send("Fake News Detection Backend Running ✅");
});

// ✅ FIX 1: endpoint is now /analyze (matches your frontend fetch call)
app.post("/analyze", (req, res) => {

    // ✅ FIX 2: reading "text" (matches what frontend sends)
    const news = req.body.text;

    if (!news) {
        return res.json({ prediction: "Please enter some content", confidence: 0 });
    }

    // ✅ FIX 3: absolute path so it works no matter where you run Node from
    const scriptPath = path.join(__dirname, "../ml_model/predict.py");

    // ✅ FIX 4: pass text via stdin instead of command-line arg
    //    This avoids crashes when the article contains quotes or special characters
    const command = `python "${scriptPath}"`;

    const child = exec(command, (error, stdout, stderr) => {
        // ✅ FIX 5: log stderr so you can actually see Python errors
        if (stderr) {
            console.error("Python stderr:", stderr);
        }

        if (error) {
            console.error("Exec error:", error.message);
            return res.json({ prediction: "Error running ML model", confidence: 0 });
        }

        const output = stdout.trim().split("|");

        // Guard: make sure output has both parts
        if (output.length < 2) {
            console.error("Unexpected output from Python:", stdout);
            return res.json({ prediction: "Unexpected model output", confidence: 0 });
        }

        const prediction  = output[0].trim();   // "Fake" or "Real"
        const confidence  = parseFloat(output[1].trim()); // e.g. 87.34

        console.log(`Prediction: ${prediction}, Confidence: ${confidence}%`);

        res.json({
            prediction: prediction,
            confidence: confidence
        });
    });

    // ✅ FIX 4 cont: write the news text into Python's stdin, then close it
    child.stdin.write(news);
    child.stdin.end();
});

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
});
