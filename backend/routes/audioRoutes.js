app.post("/upload", upload.single("audio"), (req, res) => {
  if (!req.file) {
    console.log("No file received.");
    return res.status(400).send("No file uploaded.");
  }

  console.log("File uploaded: ", req.file); // Log the file details
  res.json({
    message: "Audio file uploaded successfully!",
    file: req.file,  // Send the file information back to the frontend
  });
});
