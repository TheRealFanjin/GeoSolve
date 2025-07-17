const express = require("express");
const app = express();
app.set("view engine", "ejs");
app.get('/', (req, res) => {
    console.log("here");
    res.render("index");
})
app.listen(3000);

/*
let imageButton = document.getElementById("image");
const imageSend = () => {
    console.log(imageButton.files[0]);
    let outputButton = document.getElementById("outputImage");
    let reader = new FileReader();
    const setImage = () => {
        outputButton.src = reader.result;
        outputButton.removeAttribute("hidden");
    }
    reader.readAsDataURL(imageButton.files[0]);
    reader.addEventListener("load", setImage);
}
imageButton.addEventListener("change",imageSend,false);
*/

