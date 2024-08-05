
import { NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";
import csvtojson from "csvtojson";
import fs from "fs";
require("dotenv").config();
const axios = require('axios')
const newColumnName = 'age_group';
const newColumnName2 = 'sentiment';
let record: any;
const newColumnData: any = [];
const newColumnData2: any = [];
let age_data: any;
const result: any[] = [];
import OpenAI from "openai";
import { json } from "body-parser";
// Make sure to import OpenAI properly
const csv = require('fast-csv');
const openai = new OpenAI({ apiKey: process.env.OPENAI_SECRET_KEY }); // Make sure to import OpenAI properly
export async function POST(req: any) {
  const data = await req.formData();
  const file = data.get("file");
  const csvFilePath = `${file.name}`;
  if (!file) {
    console.log("file er");
    return NextResponse.json(
      { message: "No file found", success: false },
      { status: 500 }
    );
  }
  const byteData = await file.arrayBuffer();
  const buffer = Buffer.from(byteData);
  const filepath = `${file.name}`;
  await writeFile(filepath, buffer);
  const jsonArray = await csvtojson().fromFile(filepath);
  let textWithoutURLs: string[] = [];
  let sentimentprompts: string[] = [];
  const batchSize = 10;

  for (let j = 0; j < jsonArray.length; j++) {
    const row = jsonArray[j];
    let content = row.content;
    content = content.replace(/@/g, "");
    const textWithoutURL = content.replace(
      /(https?|ftp):\/\/[^\s/$.?#].[^\s]*/g,
      ""
    ) + ' |';

    textWithoutURLs.push(textWithoutURL);
    // console.log('textwithout urls====>>>>>>',textWithoutURLs.length);


    try {

      let sentiment_record: any
      if (textWithoutURLs.length === batchSize || j === jsonArray.length - 1) {
        age_data = await predictAgeInBatch(textWithoutURLs);
        // console.log(age_data.length,"=================>>>",i++);
        
        try{
          age_data = JSON.parse(age_data);
          
        }catch(er){
          
          console.log("response format error ==>>",age_data);
          
        }

        //  sentiment_record = JSON.parse(sentiment_data);

        try {
          for (let index = 0; index < age_data.length; index++) {
            newColumnData.push(age_data[index].age_category);
            result.push(age_data[index])
            //  newColumnData2.push(sentiment_record[index]?.sentiment);
            //  var post:any={age_group:newColumnData[index], sentiment:newColumnData2[index]};
            //  result.push(record);
          }
          
          
        } catch (error) {
          console.log(error);

        }

        textWithoutURLs = []; // Reset prompts array for the next batch
        sentimentprompts = [];
      }
    } catch (error) {
      console.log('Batch error====>', error);

    }
  }

  const rows: any = [];
  fs.createReadStream(csvFilePath)
    .pipe(csv.parse({ headers: true }))
    .on('data', (row: any) => rows.push(row))
    .on('end', () => {
      // Add a new column and populate it with data
      rows.forEach((row: any, index: any) => {


        row[newColumnName] = newColumnData[index];
        // row[newColumnName2]= newColumnData2[index];

      });

      // Write the updated data back to the original CSV file
      const ws = fs.createWriteStream(csvFilePath);
      csv.write(rows, { headers: true }).pipe(ws);

      ws.on('finish', () => {
        console.log('CSV file updated successfully!');
      });
    });
  try {
    return NextResponse.json(
      { message: result, success: false },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error during prediction:", error);
    return NextResponse.json(
      { message: "Prediction failed", success: false },
      { status: 500 }
    );
  }

}
var i=0;
async function predictAgeInBatch(textWithoutURLs: string[]) {
  let data: any 
  try {
    const chatCompletion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{
        role: "user", content: `
        - You are an age-group predictor with expertise in language, slang, and generational characteristics.
        - Your task is to analyze 10 text snippets, separated by "|", and predict the age group associated with each snippet based on sentiment.
        - Consider the following generational characteristics:
          - **Boomers** (1946-1964): optimistic, loyal to institutions, strong work ethic.
          - **Gen X** (1965-1980): independent, skeptical, adaptable to change.
          - **Millennials** (1981-1996): tech-savvy, focus on work-life balance, social and environmental impact.
          - **Gen Z** (1997-2012): digitally connected, diverse, entrepreneurial, pragmatic.
        - For each snippet, please provide a prediction in the following strict JSON format:
          {
            "age_category": "Millennials"  // Replace "Millennials" with the predicted age group
          }
        - If you're unable to predict a specific age group, please use "Unknown".
        - Enclose all predictions in a standard JSON array,strictly separated by commas and enclosed by square brackets.
        - Here are the 10 text snippets to analyze: ${textWithoutURLs}
        - Instead of relying solely on broad sentiment categories, please delve deeper into sentiment nuances. Analyze specific emotional expressions, slang, and keywords that might be indicative of different age groups. Consider sentence structure, formality, and complexity as well.
      `,
      }],
    
    });
   data= await chatCompletion.choices[0].message.content;
    return data;
  } catch (error) {
    console.error("Error during prediction:", error);
    return NextResponse.json(
      { message: "Prediction failed", success: false },
      { status: 500 }
    );
  }
}
