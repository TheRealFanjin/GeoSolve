const express = require("express");
const router = express.Router();
const multer = require("multer");

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "public/uploads/");
    },

    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const fileExtension = file.originalname.split(".").pop();
        cb(null, file.fieldname + "-" + uniqueSuffix + "." + fileExtension);
    }
});

const upload = multer({
    storage: storage
});
router.post("/upload", upload.single("inputImage"), (req, res) => {
    if(req.file) {
        res.status(200).json({
            message: "File uploaded successfully!",
            filename: req.file.filename,
            originalName: req.file.originalname,
            path: "/uploads/" + req.file.filename
        });
    }
    else {
        console.log("No file received.");
        res.status(400).json({message: "No file uploaded or an error occurred."});
    }
})
module.exports = router;