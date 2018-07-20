'use strict';

const video = document.querySelector('video');
const canvas = window.canvas = document.querySelector('canvas');
canvas.width = 480;
canvas.height = 360;

const image = document.getElementById('my_img');
const form = document.getElementById('my_form');
const button = document.querySelector('button');
button.onclick = function() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      var data = canvas.toDataURL('image/png');
      image.value = data;
  form.submit();

};



const constraints = {
  audio: false,
  video: true
};

function handleSuccess(stream) {
const video = document.querySelector('video');
const videoTracks = stream.getVideoTracks();
  video.srcObject = stream;

}

function handleError(error) {
  console.log('navigator.getUserMedia error: ', error);
}

navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);
