// ***************************
// Processing sketch for running a DMT test with facial recognition
// OpenCV is used for the detection of faces. 10x10 greyvalue grid of faces is used to train a classifier to recognise returning users.
// The UI values may have to be changed for different screens.
// This sketch is set up to work without an Arduino and Google AIY kit connected. For full functionality these are needed
// ***************************

import processing.serial.*;

import uibooster.*;
import uibooster.components.*;
import uibooster.model.*;
import uibooster.model.formelements.*;
import uibooster.utils.*;

import gab.opencv.*;
import processing.video.*;
import java.awt.*;
import org.opencv.core.Mat;

Capture video;
OpenCV opencv;

PImage photo;
PImage profile;
PImage[] faceImgs;
PImage[] faceImgsGray;

int imgW = 10;
int sensorNum = imgW*imgW; //number of sensors in use
int dataNum = 100; //number of data to show
float[] rawData = new float[sensorNum];
float[] modeArray = new float[dataNum]; //classification to show

double C = 64; //Cost: The regularization parameter of SVM
int d = sensorNum;     //Number of features to feed into the SVM
int lastPredY = -1;


String dataSetName = "faceData"; 
String[] attrNames = new String[]{"greyValue", "name"};
boolean[] attrIsNominal = new boolean[]{false, true};
Table csvData;
String user;
int Y;
boolean dataAvailable = false;


int div = 2, binDiv = 5;

String name;
String[] names = new String[10];
int[] nameID = new int[10];

boolean loggedIn = false;
boolean newLogin;

int screen = 0; // screen 0 is home, screen 1 is (new) login, screen 2 is logged in and test start
//int bg = 230; // background color
int timer;
int buffer = 0;
int recordtime = 5000; // milliseconds of data recording for new login

//ARDUINO
Serial myPort;                                 // Create object from Serial class
String val;                                    // Data received from the serial port
String lastInput;

//words used, word difficulty (card), word correctness and word label
String[] words = {"impertinent", "diadem", "picturesque", "reconnoiter", "preeminent", "opulence", "nebulous", "ensconced", "transmutation", "mortification", "myriad", "amiability", "querulous", "tautological", "sagacity", "renunciation", "qualitative", "beatitude", "ambiguity", "herculean", "immateriality", "abrogated", "imperceptibility", "rhododendron", "etymolocial", "extemporaneous", "sardonically", "miasma", "loquacious", "diametrical"}; //words in the list
int[] card = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}; //word difficulty
int[] correct = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; //word said
int [] [] labels = {{3, 4, 0}, {3, 6, 0}, {7, 8, 0}, {2, 5, 0}, {3, 0, 0}, {2, 3, 7}, {7, 8, 0}, {2, 3, 0}, {1, 5, 7}, {1, 4, 5}, {6, 9, 0}, {1, 4, 6}, {7, 8, 0}, {4, 8, 0}, {1, 2, 9}, {2, 5, 6}, {1, 4, 8}, {3, 4, 7}, {5, 8, 9}, {3, 7, 0}, {4, 6, 9}, {1, 0, 0}, {4, 9, 0}, {0, 0, 0}, {2, 6, 9}, {1, 3, 8}, {1, 2, 9}, {6, 0, 0}, {2, 5, 8}, {2, 4, 6}};

//LABEL COUNTERS
int counter1;    //counter for the 'a'-label
int counter2;    //c-label
int counter3;    //e-label
int counter4;    //i-label
int counter5;    //_i_-label
int counter6;    //ia-label
int counter7;    //u-label
int counter8;    //_u_-label
int counter9;    //y-label

//TURN AND TIME COUNTER
int count = 0;                                 //positions stored in array
int turnTimer = millis();                      //timer for checking the time taken for saying a word
int wordTotal;                                 //total number of words in the list
boolean first = true;
int skipTime = 10000;                          //time waited before word is skipped in ms
int upcomingWords = 4;                         //amount of upcoming words to show

//CSV FILE
Table table;
TableRow row;
String nameOfTest = "DMT_Test";

//TEXT File
PrintWriter overview;

// UI
int textSize = 70;                             //size of test text
PImage bg;
PImage bgClean;
int alpha=1, delta =1;

int r = 255;
int b = 255;
int change = 5;


void setup() {
  size(720, 1080);
  //fullScreen();
  //size(1440, 2160);
  bg = loadImage("RAIA_BG.png");
  bg.resize(720, 1080);
  bgClean = loadImage("RAIA_BG_clean.png");
  bgClean.resize(720, 1080);

  background(30);
  textSize(40);


  //MAKE CONTACT WITH ARDUINO (uncomment if Arduino is connected, otherwise leave unchanged to prevent error)
  //String portName = Serial.list()[4];         //change the 0 to a 1 or 2 etc. to match your port
  //myPort = new Serial(this, portName, 9600);
  //myPort.bufferUntil('\n'); 

  //CREATE CSV TABLE
  table = new Table();                       //make table in csv

  makeCSVFileRowsString(nameOfTest, words);  //make all empty rows first

  writeDataToCSVFileString(words, "word");   //add 1st column
  writeDataToCSVFileInt(card, "card");       //add 2nd column

  video = new Capture(this, 640/div, 480/div);
  opencv = new OpenCV(this, 640/div, 480/div);
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);  
  video.start();
  imageMode(CENTER);


  File Datafile = dataFile("faceData.csv");
  boolean exist = Datafile.isFile();

  if (!exist) {
    csvData = new Table();                                                  // create new dataset if there is none
    for (int i = 0; i < attrNames.length; i++) {
      csvData.addColumn(attrNames[i]);
      if (attrIsNominal[i]) csvData.setColumnType(i, Table.STRING);
      else csvData.setColumnType(i, Table.FLOAT);
    }
  } else {
    csvData = loadTable("faceData.csv", "header");     // get existing dataset to train the clasifier
    dataAvailable = true;
  }

  for (int i = 0; i < names.length; i++) {
    nameID[i] = i;
  }

  for (int i = 0; i < 5; i++) {
    text(table.getString(i, 0), width/2, height/2 - i * 110);
  }
  wordTotal = table.getRowCount();
  println("loaded " + nameOfTest + ".csv");
}


void draw() {
  opencv.loadImage(video);
  opencv.useColor();
  photo = opencv.getSnapshot();


  Rectangle[] faces = opencv.detect();


  faceImgs = new PImage[faces.length]; 
  faceImgsGray = new PImage[faces.length];


  if (loggedIn == false && screen == 0) {                                                        // 0 HOME SCREEN
    background(bgClean);
    textAlign(CENTER);
    fill(0);
    textSize(40);
    text("Hi, are you new?", width/2, height/6);
    text("Yes", width/2 - 100, height/2);
    pushStyle();
    if (dataAvailable) {
      fill(0);
    } else if (!dataAvailable) {
      fill(200);
    }
    text("No", width/2 + 100, height/2);
    popStyle();
  }

  if (loggedIn == false && newLogin == true && screen == 1) {                                    // 1 LOGIN SCREEN FOR NEW USERS

    background(bgClean);


    pushMatrix();
    translate(width/2, height/2);
    //rotate(PI/2);
    image(photo, 0, 0);
    popMatrix();

    if (millis() < timer && faces.length>0) {                                                    // while someone new is looking into the camera
      text("Hi " + name + ", please look in the camera", width/2, height/6);
      text(int((timer - millis()) / 1000), width/2, height/5);
      println(timer - millis() + " " + r + b);
      buffer = timer - millis();
      pushMatrix();
      pushStyle();
      noFill();
      stroke(r, 255, b);
      strokeWeight(3);
      for (int i = 0; i < faces.length; i++) {                                                    // for the face detected gather grey values
        translate(width/2, height/2);  
        //rotate(PI/2);
        translate(-160, -120);        
        rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);                          

        faceImgs[i] = new PImage();
        faceImgsGray[i] = new PImage();
        faceImgs[i] = photo.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        faceImgs[i].resize(100, 100);
        faceImgsGray[i] = photo.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height); 
        faceImgsGray[i].filter(GRAY);
        faceImgsGray[i].resize(10, 10);  

        for (int x = 0; x < faceImgsGray[0].width; x++) {
          for (int y = 0; y < faceImgsGray[0].width; y++) {
            int index = x + y * 10;
            color c = faceImgsGray[0].get(x, y) & 0xFF;
            rawData[index] = c;
          }
        }
      }
      popMatrix();
      popStyle();
      saveData();                                                                                 // save greyvalues to csv
    } else if (millis() > timer && faces.length>0) {
      profile = photo.get(faces[0].x, faces[0].y, faces[0].width, faces[0].height);               // create profile picture
      profile.resize(200, 200);
      profile.save("profilePicture" + name + ".jpg");
      println("Saved as: ", dataSetName+".csv");
      println("Created new login");
      loggedIn = true;
      screen = 2;
    } else if (faces.length == 0) {                                                               // stop data collection when there is no face
      text("Hi " + name + ", please look in the camera", width/2, height/6);
      text(int((timer - millis()) / 1000), width/2, height/5);
      timer = millis() + buffer;
    }
    r = r - change;
    b = b - change;
    if (r<0 || b<0) {
      change = 0;
    }
  }

  if (loggedIn == false && newLogin == false && screen == 1) {                                    // 1 LOGIN SCREEN FOR OLD USERS
    background(bgClean);

    pushMatrix();
    translate(width/2, height/2);
    //rotate(PI/2);
    image(photo, 0, 0);
    popMatrix();


    text("Hi, please look in the camera", width/2, height/6);


    for (int i = 0; i < faces.length; i++) {                                                    // for the face detected collect grey values and predict user with classifier
      pushMatrix();
      pushStyle();
      noFill();
      stroke(255 * i, 255 - 255 * i, 0);
      strokeWeight(3);

      translate(width/2, height/2);  
      //rotate(PI/2);
      translate(-160, -120);
      rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);                          

      faceImgs[i] = new PImage();
      faceImgsGray[i] = new PImage();
      faceImgs[i] = photo.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      faceImgs[i].resize(100, 100);
      faceImgsGray[i] = photo.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height); 
      faceImgsGray[i].filter(GRAY);
      faceImgsGray[i].resize(10, 10);  

      for (int x = 0; x < faceImgsGray[0].width; x++) {
        for (int y = 0; y < faceImgsGray[0].width; y++) {
          int index = x + y * 10;
          color c = faceImgsGray[0].get(x, y) & 0xFF;
          rawData[index] = c;
        }
      }
      popMatrix();
      popStyle();
    }

    double[] X = new double[d]; //Form a feature vector X;
    double[] dataToTest = new double[d];

    for (int i = 0; i < d; i++) {
      X[i] = rawData[i];
      dataToTest[i] = X[i];
    }
    int predictedY = (int) svmPredict(dataToTest); //SVMPredict the label of the dataToTest
    lastPredY = predictedY;

    pushStyle();
    if (faces.length > 0) {                                                            // ask for confirmation
      textSize(40);
      text("Are you " + names[lastPredY] + "?", width/2, 7*height/10);
      text("Yes", width/2 - 100, 8*height/10);
      text("No", width/2 + 100, 8*height/10);
    } else if (faces.length == 0) {
      text("Are you there?", width/2, 7*height/10);
    }

    if (mousePressed) {
      if (mouseX < width/2) {
        name = names[lastPredY];
        profile = loadImage("profilePicture" + name + ".jpg");
        loggedIn = true;
        screen = 2;
      } else if (mouseX > width/2 && screen == 1) {
        name = new UiBooster().showTextInputDialog("Please enter your name for manual login or ask your teacher for help.");
        profile = loadImage("profilePicture" + name + ".jpg");
        loggedIn = true;
        screen = 2;
      }
    }


    popStyle();
  }


  if (loggedIn == true && screen == 2 && dataAvailable == true) {                                                          // 2 START TEST SCREEN
    background(bgClean);
    text("Welcome " + name + ", you are now logged in", width/2, height/6);
    text("click to start test", width/2, 850);
    pushMatrix();
    translate(width/2, height/2);  
    //rotate(PI/2);
    image(profile, 0, 0);
    popMatrix();
    pushStyle();
    textSize(40);
    text(name, width/2, 7*height/10);
    popStyle();

    if (mousePressed) {
      screen = 3;
    }
  }

  if (loggedIn == true && screen == 3) {                                                        // 3 DMT Test
    background(bg);

    //WORD CARROUSEL
    if (count < 30) {
      pushStyle();
      textSize(30);
      text(count + "/30", 120, 80);             //word counter
      popStyle();
    }

    for (int word = 0; word < 1; word++) {     //print the next five words, then with every next word increase the fill value and decrease the text size
      pushStyle();
      fill(alpha);
      textSize(textSize - word * 12);

      if (count >= 30) {                       //when test has ended...
        textSize(textSize - 15);
        text("You did it " + name + "!", width/2, height/2 - word * 110); //display finish text
      } else if (count + word >= wordTotal) {
        text("", width/2, 600 - word * 110);  // error prevention for last 5 words
      } else {
        text(table.getString(count + word, 0 ), width/2, height/2 - word * 110); //show the next word
      }
      popStyle();
      if (count == 40) {                      // return to home screen after test is done
        delay(10000);
        screen = 0;
        loggedIn = false;
      }
    }

    for (int word = 1; word < upcomingWords; word++) {     //print the next five words, then with every next word increase the fill value and decrease the text size
      pushStyle();
      fill(0 + word * 60);
      textSize(textSize - 10 - word * 12);

      if (count + word >= wordTotal) {
        text("", width/2, 600 - word * 110);  // error prevention for last 5 words
      } else {
        text(table.getString(count + word, 0 ), width/2, height/2 - word * 110); //show the next word
      }
      popStyle();
    }

    //WRITE TEST OVERVIEW AND TERMINATE TEST

    if (count == 30) {                        //if the test has finished...

      for (int[] subArray : labels) { //this function is for checking how many words were said correctly within a label
        for (int INT : subArray) {
          if (INT == 1) { //if the column in this row of the label array contains a '1' that means a word was said correctly within the 'a'-label
            counter1++;
          } else if (INT == 2) {
            counter2++;
          } else if (INT == 3) {
            counter3++;
          } else if (INT == 4) {
            counter4++;
          } else if (INT == 5) {
            counter5++;
          } else if (INT == 6) {
            counter6++;
          } else if (INT == 7) {
            counter7++;
          } else if (INT == 8) {
            counter8++;
          } else if (INT == 9) {
            counter9++;
          }
        }
      }

      // Create a new file in the sketch directory
      overview = createWriter(name + "_test_overview.txt");

      writeDataToCSVFileInt(correct, "correct"); //add a column and upload what words were said
      saveTable(table, "data/"+ nameOfTest +".csv");   //save the table to csv
      println(correct);                      //print what words were said in the console

      //create overview for label a
      overview.println("label a: " + counter1 + "/8" + " correct");
      if (counter1 < 5) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter1 >= 5) {
        overview.println(name + " performed well for this label.");
      }

      overview.println(""); //add whitespace

      //create overview for label c
      overview.println("label c: " + counter2 + "/9" + " correct");
      if (counter2 < 6) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter2 >= 6) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label e
      overview.println("label e: " + counter3 + "/8" + " correct");
      if (counter3 < 5) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter3 >= 5) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label i
      overview.println("label i: " + counter4 + "/9" + " correct");
      if (counter4 < 6) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter4 >= 6) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label _i_
      overview.println("label _i_" + counter5 + "/6" + " correct");
      if (counter5 < 3) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter5 >= 3) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label ia
      overview.println("label ia: " + counter6 + "/8" + " correct");
      if (counter6 < 5) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter6 >= 5) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label u
      overview.println("label u: " + counter7 + "/7" + " correct");
      if (counter7 < 4) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter7 >= 4) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label _u_
      overview.println("label _u_: " + counter8 + "/8" + " correct");
      if (counter8 < 5) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter8 >= 5) {
        overview.println(name + " performed well for this label.");
      }

      overview.println("");

      //create overview for label y
      overview.println("label y: " + counter9 + "/7" + " correct");
      if (counter9 < 4) {
        overview.println(name + " had a large amount of the words in this label incorrect.");
        overview.println("Please focus on this label in their reading lessons.");
      } else if (counter9 >= 4) {
        overview.println(name + " performed well for this label.");
      }

      //terminate test
      println("test done");
      count = count + 10;
      overview.flush(); // Writes the remaining data to the file
      overview.close(); // Finishes the file


      //delay(10000);
      //exit(); // Stops the program


      //WORD DETECTION AND WORD CONTROL
    } else if (count < 30) {                 //if the test has not yet finished...
      if (count == 0 && first == true) {
        turnTimer = millis();                  //reset the timer
        first = false;
      }

      if (millis() > turnTimer + skipTime) {    //if 10 seconds have passed go to the next word
        println(words[count] + " skipped." + " next word"); //print that out in the console

        labels[count] [0] = 0; //change all values in this row to 0
        labels[count] [1] = 0;
        labels[count] [2] = 0;

        alpha = 1;
        delta = 1;

        turnTimer = millis();                  //reset the timer
        count = count + 1;                     //go to the next word in the list
      }

      if (millis() > turnTimer + 5300 && millis() < turnTimer + 10000) {
        println(millis() - turnTimer);
        dim();
      }

      if ("1".equals(val)) {                   //print this when word is said
        println(words[count] + " is correct!"); //print it out in the console
        correct[count] = 1;                    //save that word was said in array
        turnTimer = millis();                  //reset the timer

        count = count + 1;                     //go to the next word in the list

        print(count);
        print(wordTotal);

        delay(1000);                           //wait 1 second before checking for next word
      }
    }
  }
}



void mousePressed() {

  if (loggedIn == false && screen == 0) {
    if (mouseX < width/2) {
      NewLogin();
      timer = millis() + recordtime;
      println("Timer set, creating new login");
    } else if (mouseX > width/2 && screen == 0 && dataAvailable == true) {
      Login();
      delay(600);
    }
  }
}

void saveData() {
  for (int i=0; i < rawData.length; i++) {
    TableRow newRow = csvData.addRow();
    newRow.setFloat("greyValue", rawData[i]);
    newRow.setString("name", name);
  }
  saveTable(csvData, dataPath(dataSetName+".csv")); //save table as CSV file
  dataAvailable = true;
}


void NewLogin() {
  name = new UiBooster().showTextInputDialog("Welcome! What is your name?");
  screen = 1;
  newLogin = true;
}

void Login() {
  println("Retrieving user data");
  trainClassifier();
  screen = 1;
  newLogin = false;
}

void captureEvent(Capture c) {
  c.read();
}

void trainClassifier() {                                                            // trains the classifier with the data stored in faceData.csv
  double[] X = new double[d]; //Form a feature vector X;
  double[] dataToTrain = new double[d+1];

  csvData = loadTable("faceData.csv", "header");
  println(csvData.getRowCount() + " rows of data");

  int n=0;

  for ( int i=0; i < csvData.getRowCount() / 100; i++) {
    TableRow dataRow = csvData.getRow(i*100);
    user = dataRow.getString("name");

    if (n == 0) {
      names[0] = user;
      Y = nameID[0];
      n++;
    } 
    if (user.equals(names[n-1]) == false) {              // if user name isn't the last user name to be processed
      boolean nameExist = false;
      for (int g=0; g < n; g++) {             // check if user name is processed earlier
        if (names[g].equals(user) == true) {
          nameExist = true;
          Y = nameID[g];
        }
      }
      if (!nameExist) {
        names[n] = user;
        Y = nameID[n];
        n++;
      }
    }
    for (int q = 0; q < d; q++) {
      dataRow = csvData.getRow(i*100 + q);
      X[q] = dataRow.getInt("greyValue");
      dataToTrain[q] = X[q];
      //println(user, i*100+q, n, Y);
    }
    dataToTrain[d] = Y;
    trainData.add(new Data(dataToTrain)); //Add the dataToTrain to the trainingData collection.
  }

  println(csvData.getRowCount() / 100 + " sets of data processed");
  println(n + " users found:");
  for (int w = 0; w < n; w++) {
    println(names[w]);
  }
  type = n;
  //println(type);
  trainLinearSVC(d, C);
  if (model!=null) { 
    saveSVM_Model(sketchPath()+"/data/faceLogin.model", model);
    println("Model Saved");
  }
}

void serialEvent( Serial myPort) {
  //put the incoming data into a String - 
  //the '\n' is our end delimiter indicating the end of a complete packet
  val = myPort.readStringUntil('\n');

  //make sure our data isn't empty before continuing
  if (val != null) {
    //trim whitespace and formatting characters (like carriage return)
    val = trim(val);
  }
}

void makeCSVFileRowsString(String name, String array[]) {
  for (int i = 0; i < array.length; i++) {
    row = table.addRow();
  }
}

void writeDataToCSVFileString(String array[], String nameOfData) {
  table.addColumn(nameOfData, Table.STRING);
  for (int i = 0; i < array.length; i++) {
    table.setString(i, nameOfData, array[i]);
  }
}

void writeDataToCSVFileInt(int array[], String nameOfData) {
  table.addColumn(nameOfData, Table.INT);
  for (int i = 0; i < array.length; i++) {
    table.setInt(i, nameOfData, array[i]);
  }
}

void dim() {
  // fades the content
  alpha+=2*delta;
}
