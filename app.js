const express = require("express");
const apiRouter = require("./routes/api.js");
const app = express();
app.use("/api", apiRouter);
app.use(express.static("public"));

app.set("view engine", "ejs");
app.get('/', (req, res) => {
    console.log("here");
    res.render("index");
})
app.listen(3000);


