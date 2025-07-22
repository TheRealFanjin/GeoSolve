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