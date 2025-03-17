# Images classifier

Using a pre-trained image classifier of CNN to identify images. You can upload the images through the local web interface and the classifier will identify the image.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

## Demo

As you are running the app, you can open the browser and go to `http://localhost:5001/` to see the web interface. You can upload an image and the classifier will identify the image. For example, you can upload the image of number 9 like below:

![Number 9](./samples/9.png)

And upload the image to the web interface:

![Upload](./images/input.png)

The model will identify the image as number 9.

![Result](./images/output.png)
