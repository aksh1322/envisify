'use client'
import { useRef, useState, useEffect } from "react";
import './page.css';

// Rest of your code here

export default function CustomDropdown ({ options, selectedValue, onChange }:any)  {
    const [isOpen, setIsOpen] = useState(false);
  
    const handleOptionClick = (value:any) => {
      onChange(value);
      setIsOpen(false);
    };
  
    return (
      <div className="custom-dropdown">
        <div
          className={`dropdown-header ${isOpen ? "open" : ""}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <span>{selectedValue || "Choose a model"}</span>
          <i className={`arrow ${isOpen ? "up" : "down"}`} />
        </div>
        {isOpen && (
          <div className="options">
            {options.map((option:any) => (
              <div 
                key={option.value}
                className={`option ${selectedValue === option.value ? "selected" : ""}`}
                onClick={() => handleOptionClick(option.value)}
              >
                {option.label}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };
  