{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f7ff08b4-f05d-4e22-b69d-92d143d894fe",
   "metadata": {},
   "source": [
    "##                                                   AI Image Classifier"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "145419fe-b7ea-4a57-98a9-c1f6ff7e2c26",
   "metadata": {},
   "source": [
    "#### Task: \n",
    "To construct a CNN model which when provided with an image is able to detect if it is an AI generated image or a real image."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3d979c1-adde-4b44-ba79-78b046223b78",
   "metadata": {},
   "source": [
    "### Dataset: \n",
    "The dataset was prepared by scrapping images from the web using Selenium. Two sets of AI images and Real Images were scrapped and divided into two folders of train and test, further consisting of FAKE and REAL folders, with respective kinds of data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d010d61-6c57-4dc9-91c6-c9e33dc55087",
   "metadata": {},
   "source": [
    "### Importing Libraries and Unzipping File for Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3ce128c8-26a9-4887-a74f-8d4099d2da05",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "from PIL import Image\n",
    "import time\n",
    "from tempfile import TemporaryDirectory"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1874e5ce-f315-46aa-ae68-80e9742e79a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "535b649b-3e56-4579-ba39-f19ca082c55f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (2.16.1)\n",
      "Requirement already satisfied: tensorflow-intel==2.16.1 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow) (2.16.1)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.1.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=23.5.26 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (24.3.25)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.5.4)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=3.10.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.11.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (18.1.1)\n",
      "Requirement already satisfied: ml-dtypes~=0.3.1 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.3.2)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: packaging in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (23.1)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.20.3)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.31.0)\n",
      "Requirement already satisfied: setuptools in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (68.2.2)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.4.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (4.9.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.64.0)\n",
      "Requirement already satisfied: tensorboard<2.17,>=2.16 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.16.2)\n",
      "Requirement already satisfied: keras>=3.0.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.3.3)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.31.0)\n",
      "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in c:\\anaconda1\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.26.4)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\anaconda1\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.16.1->tensorflow) (0.41.2)\n",
      "Requirement already satisfied: rich in c:\\anaconda1\\lib\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (13.3.5)\n",
      "Requirement already satisfied: namex in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.0.8)\n",
      "Requirement already satisfied: optree in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.11.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\anaconda1\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\anaconda1\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\anaconda1\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\anaconda1\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (2024.2.2)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\anaconda1\\lib\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\anaconda1\\lib\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (2.2.3)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\anaconda1\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (2.1.3)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\anaconda1\\lib\\site-packages (from rich->keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\anaconda1\\lib\\site-packages (from rich->keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\anaconda1\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich->keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.1.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2fbf0fe4-e0d1-4099-b299-cdc0a0009f10",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torch in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (2.3.0)\n",
      "Requirement already satisfied: filelock in c:\\anaconda1\\lib\\site-packages (from torch) (3.13.1)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in c:\\anaconda1\\lib\\site-packages (from torch) (4.9.0)\n",
      "Requirement already satisfied: sympy in c:\\anaconda1\\lib\\site-packages (from torch) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\anaconda1\\lib\\site-packages (from torch) (3.1)\n",
      "Requirement already satisfied: jinja2 in c:\\anaconda1\\lib\\site-packages (from torch) (3.1.3)\n",
      "Requirement already satisfied: fsspec in c:\\anaconda1\\lib\\site-packages (from torch) (2023.10.0)\n",
      "Requirement already satisfied: mkl<=2021.4.0,>=2021.1.1 in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from torch) (2021.4.0)\n",
      "Requirement already satisfied: intel-openmp==2021.* in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from mkl<=2021.4.0,>=2021.1.1->torch) (2021.4.0)\n",
      "Requirement already satisfied: tbb==2021.* in c:\\users\\diya sivaprasad\\appdata\\roaming\\python\\python311\\site-packages (from mkl<=2021.4.0,>=2021.1.1->torch) (2021.12.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\anaconda1\\lib\\site-packages (from jinja2->torch) (2.1.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\anaconda1\\lib\\site-packages (from sympy->torch) (1.3.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fb45815f-59f7-44c7-b5f1-fe113a4b8ea0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow\n",
    "import torch\n",
    "import zipfile\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1abdf31d-e008-4f9f-8ecc-e0b61e1dc2a2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3e020d50-c1bc-49c6-9608-b60acbab5d5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\dell\\Project_Image_classifier\n"
     ]
    }
   ],
   "source": [
    "cwd = os.getcwd()\n",
    "print(cwd)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3ee530b9-2f04-47e5-a1f2-3db2ce5c1e78",
   "metadata": {},
   "outputs": [],
   "source": [
    "zip_file_path  = 'Images.zip'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "47d089ba-fcc9-4b26-8883-ac0c0a956aa4",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images, train_labels = [], []\n",
    "test_images, test_labels = [], [] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "faef700b-a6f1-4757-9c00-e5c40875c964",
   "metadata": {},
   "outputs": [],
   "source": [
    "import zipfile"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6d4a9559-4699-471f-9c92-043bfaaaaf4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def is_image_file(filename):\n",
    "    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "73b3ecf2-a231-4a4c-9279-fb7da410bf15",
   "metadata": {},
   "outputs": [],
   "source": [
    "with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:\n",
    "    for file_info in zip_ref.infolist():\n",
    "        file_path = file_info.filename\n",
    "\n",
    "        if not is_image_file(file_path):\n",
    "            continue\n",
    "        \n",
    "        if 'train' in file_path:\n",
    "            folder = 'train'\n",
    "        elif 'test' in file_path:\n",
    "            folder = 'test'\n",
    "\n",
    "            \n",
    "        if 'REAL' in file_path:\n",
    "            label = 0\n",
    "        elif 'FAKE' in file_path:\n",
    "            label = 1\n",
    "            \n",
    "        with zip_ref.open(file_path) as file:\n",
    "            image = Image.open(file)\n",
    "            image = image.convert('RGB')\n",
    "        \n",
    "        # Append the image and label to the respective lists\n",
    "        if folder == 'train':\n",
    "            train_images.append(image)\n",
    "            train_labels.append(label)\n",
    "        elif folder == 'test':\n",
    "            test_images.append(image)\n",
    "            test_labels.append(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f09b2797-fb7c-4040-b057-7f15f65b8dc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Images =  4000\n",
      "Testing Images =  400\n"
     ]
    }
   ],
   "source": [
    "print(\"Training Images = \", len(train_images))\n",
    "print(\"Testing Images = \", len(test_images))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6bd158e5-35d6-4cf2-8fc1-730df869351e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4000\n",
      "400\n"
     ]
    }
   ],
   "source": [
    "print(len(train_labels))\n",
    "print(len(test_labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23c5b83d-9f5b-4f71-891f-9f5d2be9d161",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "add49568-f140-4059-8d15-300f28e5d79d",
   "metadata": {},
   "source": [
    "### Preparing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7bf856cc-187a-4013-a2c5-c7438351d96d",
   "metadata": {},
   "outputs": [],
   "source": [
    "image_size = (150, 150)\n",
    "batch_size = 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5631ddb7-c97c-45dd-8af4-935b5cd0d8d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d594f9a7-7d6e-4947-8f94-5809fac95c1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4e47bf3e-1fce-4cd6-ae18-8956ecf64249",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(int(len(train_images))):\n",
    "    X_train.append(img_to_array(train_images[i].resize(image_size)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8472507f-881f-45b8-b253-ef21e9b6ad98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4000"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f940b03d-7de5-4682-a318-53123e76e318",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = np.array(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e4923994-532c-42e8-9b78-99d04eff2fae",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "157f436a-6b9d-4661-8b95-d0101ae5ff81",
   "metadata": {},
   "outputs": [],
   "source": [
    "i = 0;\n",
    "for i in range(int(len(test_images))):\n",
    "    X_test.append(img_to_array(test_images[i].resize(image_size)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "8c8ccb3a-8186-4a5b-bb85-ed967583f7c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = np.array(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "1e60d1f1-6c69-4bad-a020-8dbe5c336cac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total training images =  4000\n",
      "Total testing images =  400\n"
     ]
    }
   ],
   "source": [
    "print(\"Total training images = \", len(X_train))\n",
    "print(\"Total testing images = \", len(X_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c11de08c-cac3-4245-96b7-ed6dee65cf41",
   "metadata": {},
   "source": [
    "We have 4000 images to train the model and 400 images to test the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "c5798dfc-0c3d-450c-8e07-fc622b3a9cb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#creating the y_train and y_test data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8d67b690-9e51-4624-8ae0-6767cf5ec2f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = []\n",
    "y_test = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8a758527-b25e-4c0e-8631-801389340518",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(int(len(train_labels))):\n",
    "    y_train.append(train_labels[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "2ac47dcc-053c-4185-bf82-4e30a80eb1b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(int(len(test_labels))):\n",
    "    y_test.append(test_labels[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "8e228f64-514e-4cef-babc-0b56a1a3043f",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test = np.array(y_test)\n",
    "y_train = np.array(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94aa1989-33a9-4202-a781-e2e2ccfc1024",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "0c13f3e4-d58e-4738-b54e-594e8874a439",
   "metadata": {},
   "outputs": [],
   "source": [
    "#normalise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c3f3be78-c0f7-4398-a6d2-626fb7ddd986",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = X_train/255.0\n",
    "X_test = X_test/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "311a8d98-4570-4978-b52d-81e3f8ec0325",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "size of y_train =  4000\n",
      "size of y_test =  400\n"
     ]
    }
   ],
   "source": [
    "print(\"size of y_train = \", len(y_train))\n",
    "print(\"size of y_test = \", len(y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "731b121e-769c-43eb-a522-afd5e1aa9ffb",
   "metadata": {},
   "source": [
    "### CNN Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b9f5e5d3-1733-45a7-9ff5-07fdff3085c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "9bc578a0-e201-496b-a631-25776a9cb587",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "3b5e195f-b326-48a1-9e71-831abe7212cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "ffbd8823-02c6-4d93-89d6-28ea15a0c3aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\dell\\anaconda3\\Anaconda\\Lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "model.add(Conv2D(16, (4, 4), 1, activation='relu', input_shape=(150, 150, 3)))\n",
    "model.add(MaxPooling2D())\n",
    "\n",
    "model.add(Conv2D(32, (4, 4), 1, activation='relu'))\n",
    "model.add(MaxPooling2D())\n",
    "\n",
    "model.add(Conv2D(16, (4, 4), 1, activation='relu'))\n",
    "model.add(MaxPooling2D())\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(32, activation='relu'))\n",
    "model.add(Dense(1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "0be7a068-0f0f-4149-8652-71097be82e56",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "e8960a14-099e-4fc9-a22c-13491ab3c185",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">147</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">147</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)        │             <span style=\"color: #00af00; text-decoration-color: #00af00\">784</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">73</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">73</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">70</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">70</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │           <span style=\"color: #00af00; text-decoration-color: #00af00\">8,224</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">35</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">35</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)          │           <span style=\"color: #00af00; text-decoration-color: #00af00\">8,208</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)                │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)                  │         <span style=\"color: #00af00; text-decoration-color: #00af00\">131,104</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>)                   │              <span style=\"color: #00af00; text-decoration-color: #00af00\">33</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m147\u001b[0m, \u001b[38;5;34m147\u001b[0m, \u001b[38;5;34m16\u001b[0m)        │             \u001b[38;5;34m784\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m73\u001b[0m, \u001b[38;5;34m73\u001b[0m, \u001b[38;5;34m16\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m70\u001b[0m, \u001b[38;5;34m70\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │           \u001b[38;5;34m8,224\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m35\u001b[0m, \u001b[38;5;34m35\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m16\u001b[0m)          │           \u001b[38;5;34m8,208\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m16\u001b[0m, \u001b[38;5;34m16\u001b[0m, \u001b[38;5;34m16\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)                │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m)                  │         \u001b[38;5;34m131,104\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m)                   │              \u001b[38;5;34m33\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">148,353</span> (579.50 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m148,353\u001b[0m (579.50 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">148,353</span> (579.50 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m148,353\u001b[0m (579.50 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "7dd3ea88-4dac-4261-abfa-37381379bba2",
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs = 15\n",
    "batch_size = 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "047fef1a-9596-4bd9-a7a9-d82ea91d3923",
   "metadata": {},
   "outputs": [],
   "source": [
    "logdir = 'logs'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "89b8a579-9605-4217-b7c2-9bff932d1030",
   "metadata": {},
   "outputs": [],
   "source": [
    "tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir = logdir)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14e8d079-66e1-40db-96c3-6a1773d57a49",
   "metadata": {},
   "source": [
    "### Model Training "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "4e2fbb64-a09b-42fc-b11b-39648c924800",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m58s\u001b[0m 227ms/step - accuracy: 0.5206 - loss: 0.6888\n",
      "Epoch 2/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22s\u001b[0m 266ms/step - accuracy: 0.6954 - loss: 0.5728\n",
      "Epoch 3/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 228ms/step - accuracy: 0.7733 - loss: 0.4759\n",
      "Epoch 4/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 213ms/step - accuracy: 0.7915 - loss: 0.4597\n",
      "Epoch 5/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 205ms/step - accuracy: 0.8183 - loss: 0.3936\n",
      "Epoch 6/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 206ms/step - accuracy: 0.8232 - loss: 0.3838\n",
      "Epoch 7/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 210ms/step - accuracy: 0.8357 - loss: 0.3695\n",
      "Epoch 8/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 201ms/step - accuracy: 0.8416 - loss: 0.3542\n",
      "Epoch 9/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 202ms/step - accuracy: 0.8501 - loss: 0.3412\n",
      "Epoch 10/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 198ms/step - accuracy: 0.8797 - loss: 0.3061\n",
      "Epoch 11/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 197ms/step - accuracy: 0.8735 - loss: 0.2989\n",
      "Epoch 12/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 200ms/step - accuracy: 0.8870 - loss: 0.2760\n",
      "Epoch 13/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 212ms/step - accuracy: 0.9000 - loss: 0.2481\n",
      "Epoch 14/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 213ms/step - accuracy: 0.9085 - loss: 0.2193\n",
      "Epoch 15/15\n",
      "\u001b[1m80/80\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 224ms/step - accuracy: 0.8989 - loss: 0.2472\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(X_train, y_train, epochs = epochs,batch_size = batch_size, callbacks=[tensorboard_callback])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "654176be-ea66-489e-820d-3c61c502f90c",
   "metadata": {},
   "source": [
    "### Model Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "5512838b-3dad-4251-aa96-6979e3efad86",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 39ms/step - accuracy: 0.6857 - loss: 0.6332\n"
     ]
    }
   ],
   "source": [
    "loss, accuracy  = model.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "e4cfef42-ac1e-4e0c-83af-4d092c13ca4e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy on test data =  76.2499988079071 %\n"
     ]
    }
   ],
   "source": [
    "print(\"accuracy on test data = \", accuracy*100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "35ca640a-8984-402e-ad01-7077e88c4cb1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m125/125\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 40ms/step - accuracy: 0.8925 - loss: 0.2566\n"
     ]
    }
   ],
   "source": [
    "train_loss, train_accuracy = model.evaluate(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "466c9619-0370-43d6-b50d-f95a4bfeb7ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy on training data =  91.44999980926514 %\n"
     ]
    }
   ],
   "source": [
    "print(\"accuracy on training data = \", train_accuracy*100, \"%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f7e839e-1d52-4cfd-93b4-7abb01e8afcc",
   "metadata": {},
   "source": [
    "### Plotting Accuracy and Loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "08a7c167-ff4f-4878-997c-2552db82aae9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAHNCAYAAADMjHveAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABbg0lEQVR4nO3deVxU9f7H8dfIjgLuCIpLbrnlAuWWaVl41UyzUiu31MqblmaZmr+svLcor6ZlaWlpm6a3LK+ppVTuthhii5pbKC4YboErCJzfH18BR1ABgQPD+/l4nAczZ86Z+cxYzJvv+S4Oy7IsRERERGxSyu4CREREpGRTGBERERFbKYyIiIiIrRRGRERExFYKIyIiImIrhRERERGxlcKIiIiI2EphRERERGylMCIiIiK2UhgRkXzRoUMHRo4caXcZIlIMKYyIFIKBAwfSo0cPu8soEs6ePUu5cuUoX748Z8+etbscESkCFEZEpFAtWrSIxo0b07BhQz7//HNba7Esi5SUFFtrEBGFEZEiYc2aNdx00014eXkRFBTE2LFjnb4kP/vsM5o0aYKPjw8VKlTg9ttv5/Tp0wCsXr2am266idKlS1O2bFnatm3Lvn37LvtaY8aMoV69evj6+nLdddfx3HPPcf78+YzHX3jhBZo1a8ZHH31EzZo1CQgIoE+fPpw8eTLjmNOnT9O/f3/KlClDUFAQU6ZMyfF7fe+99+jbty99+/blvffey/L41q1b6dq1K/7+/vj5+dGuXTv27NmT8ficOXNo1KhRxmc1fPhwAPbu3YvD4WDLli0Zx/799984HA5Wr16d8Vk5HA5WrFhBWFgYXl5erFu3jj179tC9e3cCAwMpU6YMN954I998841TXUlJSTzzzDOEhITg5eVF3bp1ee+997Asizp16jB58mSn43///XdKlSrlVLuIZE9hRMRmBw8epEuXLtx444388ssvzJw5k/fee49///vfAMTFxXH//fczaNAgtm/fzurVq+nZs2fGX/U9evSgffv2/Prrr3z//fc88sgjOByOy76en58f77//Ptu2beP1119n9uzZTJ061emYPXv2sHjxYpYuXcrSpUtZs2YNr7zySsbjo0ePZtWqVXzxxResXLmS1atXExUVddX3umfPHr7//nt69epFr1692LhxI3/++afTZ3HLLbfg7e3Nd999R1RUFIMG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BCZNmmR3OSLFjlpGRERExFZqGRERERFbKYyIiIiIrRRGRERExFYKIyIiImIrhRERERGxlcKIiIiI2EphRERERGylMCIiIiK2UhgRERERWymMiIiIiK0URkRERMRWCiMiIiJiK4URERERsZXCiIiIiNhKYURERERspTAiIiIitlIYEREREVspjIiIiIitFEZERETEVgojIiIiYiuFEREREbGVwoiIiIjYSmFEREREbKUwIiIiIrZSGBERERFbKYyIiIiIrRRGRERExFYKIyIiImIrhRERERGxlcKIiIiI2EphRERERGylMCIiIiK2cre7gJxIS0vj0KFD+Pn54XA47C5HREREcsCyLE6ePElwcDClSl2+/aNYhJFDhw4REhJidxkiIiKSB/v376datWqXfbxYhBE/Pz/AvBl/f3+bqxEREZGcSExMJCQkJON7/HKKRRhJvzTj7++vMCIiIlLMXK2LhTqwioiIiK0URkRERMRWCiMiIiJiq2LRZyQnLMsiJSWF1NRUu0uRXHBzc8Pd3V1DtkVESjCXCCPJycnExcVx5swZu0uRPPD19SUoKAhPT0+7SxERERsU+zCSlpZGTEwMbm5uBAcH4+npqb+yiwnLskhOTubIkSPExMRQt27dK06KIyIirqnYh5Hk5GTS0tIICQnB19fX7nIkl3x8fPDw8GDfvn0kJyfj7e1td0kiIlLIXObPUP1FXXzp305EpGTTt4CIiIjYSmHEJh06dGDkyJF2lyEiImI7hRERERGxlcKIiIiI2EphpAg4ceIE/fv3p1y5cvj6+tK5c2d27dqV8fi+ffvo1q0b5cqVo3Tp0jRq1Ijly5dnnPvggw9SqVIlfHx8qFu3LnPnzrXrrYiISE6lnIa/1sC2SfDbi3Dgf3DmAFiW3ZUVujwN7Z0xYwb/+c9/iIuLo1GjRkybNo127dpd9vi33nqLN998k71791K9enXGjx9P//7981z0lVgW2DX3ma8v5GWKk4EDB7Jr1y6WLFmCv78/Y8aMoUuXLmzbtg0PDw+GDRtGcnIya9eupXTp0mzbto0yZcoA8Nxzz7Ft2za++uorKlasyO7duzl79mw+vzMREbkmVhok7oCjP8CxH83PhN/BymbWcK9KUD4UyrcwP8u1gNI18vYFU0zkOowsXLiQkSNHMmPGDNq2bcs777xD586d2bZtG9WrV89y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15OeiYiIiOQnLZQnIiIitlIYEREREVspjIiIiIitFEZERETEViU6jMyYMYNatWrh7e1NaGgo69ats7ukQhEREcGNN96In58flStXpkePHuzYscPusmwTERGBw+Fg5MiRdpdSqA4ePEjfvn2pUKECvr6+NGvWjKioKLvLKjQpKSn83//9H7Vq1cLHx4frrruOiRMnkpaWZndpBWLt2rV069aN4OBgHA4HixcvdnrcsixeeOEFgoOD8fHxoUOHDmzdutWeYgvIlT6D8+fPM2bMGJo0aULp0qUJDg6mf//+HDp0yL6C89nV/hu42KOPPorD4WDatGmFUluJDSMLFy5k5MiRjB8/nujoaNq1a0fnzp2d5khxVWvWrGHYsGH88MMPREZGkpKSQnh4OKdPn7a7tEK3adMmZs2axQ033GB3KYXqxIkTtG3bFg8PD7766iu2bdvGlClTCnSm46Lm1Vdf5e233+bNN99k+/btTJo0if/85z9Mnz7d7tIKxOnTp2natClvvvlmto9PmjSJ1157jTfffJNNmzZRpUoV7rjjDk6ePFnIlRacK30GZ86cYfPmzTz33HNs3ryZzz//nJ07d3LXXXfZUGnBuNp/A+kWL17Mjz/+eNUp3POVVULddNNN1tChQ532XX/99dbYsWNtqsg+8fHxFmCtWbPG7lIK1cmTJ626detakZGRVvv27a0RI0bYXVKhGTNmjHXzzTfbXYatunbtag0aNMhpX8+ePa2+ffvaVFHhAawvvvgi435aWppVpUoV65VXXsnYd+7cOSsgIMB6++23baiw4F36GWTnp59+sgBr3759hVNUIbrc+z9w4IBVtWpV6/fff7dq1KhhTZ06tVDqKZEtI8nJyURFRREeHu60Pzw8nI0bN9pUlX0SEhIAKF++vM2VFK5hw4bRtWvXqy7c6IqWLFlCWFgY9913H5UrV6Z58+bMnj3b7rIK1c0338y3337Lzp07Afjll19Yv349Xbp0sbmywhcTE8Phw4edfid6eXnRvn37Evk7MV1CQgIOh6PEtBimpaXRr18/Ro8eTaNGjQr1tfM0A2txd/ToUVJTU7OsNBwYGJhlRWJXZ1kWo0aN4uabb6Zx48Z2l1NoFixYwObNm9m0aZPdpdjizz//ZObMmYwaNYpnn32Wn376iSeeeAIvLy/69+9vd3mFYsyYMSQkJHD99dfj5uZGamoqL730Evfff7/dpRW69N972f1O3Ldvnx0l2e7cuXOMHTuWBx54oMQsnvfqq6/i7u7OE088UeivXSLDSDqHw+F037KsLPtc3fDhw/n1119Zv3693aUUmv379zNixAhWrlyJt7e33eXYIi0tjbCwMF5++WUAmjdvztatW5k5c2aJCSMLFy7k448/Zv78+TRq1IgtW7YwcuRIgoODGTBggN3l2UK/E43z58/Tp08f0tLSmDFjht3lFIqoqChef/11Nm/ebMu/eYm8TFOxYkXc3NyytILEx8dn+cvAlT3++OMsWbKEVatWUa1aNbvLKTRRUVHEx8cTGhqKu7s77u7urFmzhjfeeAN3d3dSU1PtLrHABQUF0bBhQ6d9DRo0KBEduNONHj2asWPH0qdPH5o0aUK/fv148skniYiIsLu0QlelShWAEv87EUwQ6dWrFzExMURGRpaYVpF169YRHx9P9erVM34v7tu3j6eeeoqaNWsW+OuXyDDi6elJaGgokZGRTvsjIyNp06aNTVUVHsuyGD58OJ9//jnfffcdtWrVsrukQtWxY0d+++03tmzZkrGFhYXx4IMPsmXLFtzc3OwuscC1bds2y3DunTt3Zqy+XRKcOXOGUqWcfwW6ubm57NDeK6lVqxZVqlRx+p2YnJzMmjVrSsTvxHTpQWTXrl188803VKhQwe6SCk2/fv349ddfnX4vBgcHM3r0aFasWFHgr19iL9OMGjWKfv36ERYWRuvWrZk1axaxsbEMHTrU7tIK3LBhw5g/fz7/+9//8PPzy/hrKCAgAB8fH5urK3h+fn5Z+seULl2aChUqlJh+M08++SRt2rTh5ZdfplevXvz000/MmjWLWbNm2V1aoenWrRsvvfQS1atXp1GjRkRHR/Paa68xaNAgu0srEKdOnWL37t0Z92NiYtiyZQvly5enevXqjBw5kpdffpm6detSt25dXn75ZXx9fXnggQdsrDp/XekzCA4O5t5772Xz5s0sXbqU1NTUjN+N5cuXx9PT066y883V/hu4NHx5eHhQpUoV6tevX/DFFcqYnSLqrbfesmrUqGF5enpaLVq0KDFDW4Fst7lz59pdmm1K2tBey7KsL7/80mrcuLHl5eVlXX/99dasWbPsLqlQJSYmWiNGjLCqV69ueXt7W9ddd501fvx4Kykpye7SCsSqVauy/f9+wIABlmWZ4b3PP/+8VaVKFcvLy8u65ZZbrN9++83eovPZlT6DmJiYy/5uXLVqld2l54ur/TdwqcIc2uuwLMsq+MgjIiIikr0S2WdEREREig6FEREREbGVwoiIiIjYSmFEREREbKUwIiIiIrZSGBERERFbKYyIiIiIrRRGRERExFYKIyIiImIrhRERERGxlcKIiIiI2EphRERERGz1/5/7iW+Lj2/NAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "plt.plot(history.history['loss'], color='blue', label='loss')\n",
    "plt.plot(history.history['accuracy'], color='orange', label='accuracy')\n",
    "fig.suptitle('Loss and Accuracy', fontsize=10)\n",
    "plt.legend(loc=\"upper left\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "545d6978-ab54-487b-ac57-2f73610ccbfc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "50a46e49-680f-4138-94f8-43eb1a270115",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 51ms/step\n"
     ]
    }
   ],
   "source": [
    "y_result = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "5b7137e0-08da-4c6e-ab16-55df86a5164e",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(y_result)):\n",
    "    if(y_result[i] > 0.5):\n",
    "        y_result[i] = 1\n",
    "    else:\n",
    "        y_result[i] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "8301621f-ff40-4dc0-b053-0eb8552468b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision Score :  0.625\n",
      "Accuracy Score :  0.7625\n",
      "F1 score :  0.7246376811594203\n",
      "ROC AUC Score :  0.7839756592292089\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision Score : \", precision_score(y_result, y_test))\n",
    "print(\"Accuracy Score : \", accuracy_score(y_result, y_test))\n",
    "print(\"F1 score : \", f1_score(y_result, y_test))\n",
    "print(\"ROC AUC Score : \", roc_auc_score(y_result, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b7d9e46f-174e-40d4-a71b-3be1096efe8b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix :  [[180  20]\n",
      " [ 75 125]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix : \", confusion_matrix(y_test, y_result))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "460050ac-c678-4116-888a-6d214f12776d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "4740bef5-fa47-4b5a-83e0-76f6ae8450aa",
   "metadata": {},
   "source": [
    "### Saving the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "cfcdf50d-59ac-4aa4-97b4-bbb405e396ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "b544f19d-9f5c-4715-9413-2a5016e7964c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "model.save(os.path.join('model','ai_imageclassifier.h5'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ba3d6ac-ce0b-41cc-a8db-ad13614bc02b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "d0ba35cb-069d-4061-9c0a-e8704b204ae9",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "load_model() takes 0 positional arguments but 1 was given",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[78], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m model \u001b[38;5;241m=\u001b[39m load_model(os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmodel\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mai_imageclassifier.h5\u001b[39m\u001b[38;5;124m'\u001b[39m))\n",
      "\u001b[1;31mTypeError\u001b[0m: load_model() takes 0 positional arguments but 1 was given"
     ]
    }
   ],
   "source": [
    "model = load_model(os.path.join('model','ai_imageclassifier.h5'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5cacbc9-9599-48c2-be9b-4b28db31557c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "879eb09e-cc30-49bc-aabd-4084b264a012",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55185f23-633b-47d9-b640-451ba85e57d7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a92084a6-90f6-47f8-a3b4-416c8be0dd63",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
