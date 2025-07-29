const express = require("express");
const apiRouter = require("./routes/upload.js");
const app = express();

app.use("/", apiRouter);
app.use(express.static("public"));

app.set("view engine", "ejs");
app.get('/', (req, res) => {
    res.render("index");
})
app.listen(3000);


