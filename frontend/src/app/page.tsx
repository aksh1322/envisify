
"use client";
import Image from "next/image";
import styles from "./page.module.css";
import Head from "next/head";
import Header from "./header/page";
import Footer from "./footer/page";
import { useRef, useState, useEffect } from "react";
import FileUploadIcon from "public/icons8-upload-48.png";
import Papa from "papaparse";
import CustomDropdown from "./customdropdown/page";

export default function Home() {
  const [jsonData, setJsonData] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileData, setFileData] = useState<Blob | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [uploadButtonText, setUploadButtonText] = useState<string>("Upload");
  useEffect(() => {
    // Fetch available models from the backend API
    fetch("https://mlstatus.envisifygi.com/api/models")
      .then(response => response.json())
      .then(data => {
        setAvailableModels(data);
      })
      .catch(error => console.error("Error fetching models:", error));
  }, []);
  const showLoader = () => {
    setLoading(true);
  };
  const resetFile = () => {
    setSelectedFile(null);
    setJsonData("");
    setFileData(null);
    setUploadButtonText("Upload");
  };
  const [selectedModel, setSelectedModel] = useState<string>("");
  const hideLoader = () => {
    setLoading(false);
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      resetFile();
      return;
    }

    setSelectedFile(file);
    setJsonData(file.name);
    setFileData(null);
    setUploadButtonText("Process");
  };
  const handleCloseIconClick = () => {
    resetFile();
  };
  const handleModelChange = (model: string) => {
    console.log('in handle change');

    // const model = e.target?.value;

    console.log(model);

    setSelectedModel(model);
    const selectedModelInfo: any = availableModels.find((m: any) => m.value === model);

    if (selectedModelInfo) {
      setJsonData(selectedModelInfo.description);
    }

  };
  const handleUploadClick = async () => {
    if (!selectedFile) {
      alert("Please select a file.");
      return;
    }
    if (!selectedModel) {
      alert("Please choose a model.");
      return;
    }
    showLoader();
    const data = new FormData();
    data.set("file", selectedFile);

    data.set("model", selectedModel);
    try {
      // https://mlstatus.envisifygi.com
      const response = await fetch(`https://mlstatus.envisifygi.com/api/inference/${selectedModel}`, {
        method: "POST",
        body: data,
      });

      if (response.ok) {
        const fileBlob = await response.blob();
        setFileData(fileBlob);
      } else {
        console.error("Error:", response.status, response.statusText);
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      hideLoader();
    }
  };

  const handleDownloadClick = () => {
    if (fileData) {
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(fileData);
      link.download = `${jsonData.toLowerCase().replace('.csv','')}_${selectedModel}_result.csv`;

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      // Refresh the page
      window.location.reload();
    }
  };

  return (
    <>
      <img id="logo" src="logo.png" alt="Logo"></img>
      <div id="content-div">
        <label htmlFor="model" id="model_heading"  ><b>PLEASE CHOOSE A MODEL</b></label>
        <CustomDropdown
          options={availableModels}
          selectedValue={selectedModel}
          onChange={handleModelChange}
        />
        {availableModels.map((model: any) => (
          selectedModel === model.value && (
            <div
              key={model.value}
              className="textarea"
            
            >
             <p style={{margin:'0'}}>{model.description}</p>
            </div>
          )
        ))}
        <div style={{marginTop:'60px'}}>
          <span style={{ position: 'relative', bottom: '-8px', left: '1%' ,marginTop:'80px'}}>Upload CSV File</span>
          
          <div>
            <input
              type="file"
              id="csvFile"
              accept=".csv"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <button
              className="upload-btn"
              onClick={() => uploadButtonText == 'Upload' ?fileInputRef.current?.click():handleUploadClick()}
            >  <p className="upload-div">
                <Image
                  title="upload"
                  src="/icons8-upload-48.png"
                  alt="Upload Icon"
                  width={24}
                  height={24}
                />{' '}
                <span style={{ position: 'relative', top: '-5px' }}>{uploadButtonText} </span>{' '}

              </p>
            </button>

            {selectedFile &&

              (
                <div style={{ marginTop: '29px' }}>
                  <Image
                    
                    src="document.svg"
                    alt="File Icon"
                    width={16}
                    height={16}
                    style={{ position: 'relative', top: '3px' }}
                  />
                  <span style={{ position: 'relative', marginLeft: '5px' }}>

                    {selectedFile.name}
                  </span>
                 
                  <Image
                    title="unselect file"
                    src="/close.png"
                    alt="remove file"
                    width={16}
                    height={16}
                     style={{ position: 'relative', marginLeft:'45px' ,top:'3px' }}
                     onClick={handleCloseIconClick}
                  />
                </div>

              )}
            
          </div>
          <div className="loader" style={{ display: loading ? 'block' : 'none' }}>
            <div className="justify-content-center jimu-primary-loading">

            </div>
          </div>

          {fileData && (
            <div className={styles.message} style={{ color: 'green', fontSize: '15px', marginTop: "7%", }}>
              File is ready for download. Click the &quot;Download&quot; button below.
            </div>
          )}
          {fileData && (
            <button
              className="download-btn"
              onClick={handleDownloadClick}
            >
              Download
            </button>
          )}
        </div>
      </div>
    </>
  );
}

