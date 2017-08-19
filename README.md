# Tensorflow object detector web service

Rudimentary flask web service that wraps the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection).

## Instructions

Build the docker image:

```sh
docker-compose build
```

Start the service:

```sh
docker-compose up -d
```

This starts the web service at 'localhost:5000' and some jupyter service at 'localhost:8888' (to get the exact location enter `docker-compose logs app` and search the logs for the jupyter url).

Apply object detection on a random cat:
[http://localhost:5000/detect_objects](http://localhost:5000/detect_objects)

<p align="center">
  <img src="doc/cat_detections_output.jpg">
</p>


You can also apply object detection to your own given image:

[http://localhost:5000/detect_objects?url=$YOUR_URL](http://localhost:5000/detect_objects?url=$YOUR_URL)


