import React from "react";
import FileUpload from "./components/FileUpload";
import { HeroGeometric } from "./components/Landing";
import { BrowserRouter, Route, Routes } from "react-router-dom";

function App() {
  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HeroGeometric />} />
          <Route path="/upload" element={<FileUpload />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
