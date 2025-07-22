const express = require("express");
const router = express.Router();
router.post("/upload", (req, res) => {
    res.sendStatus(200);
    console.log("upload");
});
module.exports = router;