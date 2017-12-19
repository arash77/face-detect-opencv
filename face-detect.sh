#!/bin/bash

echo "welcome to face detection installer!"
echo "this project is created by ARASH KADKHODAEI (arash77.ir)"

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install zbar-tools
sudo apt-get install python-zbar
sudo apt-get install libzbar0
sudo apt-get install libv4l-dev
sudo apt-get install libperl-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install python-gobject-dev
sudo apt-get install python-qrtools
sudo apt-get install libzbar-dev
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev libgstreamer0.10-0-dbg libgstreamer0.10-0 libgstreamer0.10-dev libv4l-0 libv4l-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python-numpy python-scipy python-matplotlib
sudo apt-get install default-jdk ant
sudo apt-get install libgtkglext1-dev
sudo apt-get install v4l-utils
sudo apt-get install python2.7-dev python3-dev
sudo apt-get install python-picamera
sudo apt-get install python-pip
sudo pip install zbar
sudo pip install numpy
sudo pip install telepot

wget facedetect.zip
sudo unzip -d face-detect/ facedetect.zip
rm facedetect.zip
crontab -l > autocron
echo "@reboot sh /home/pi/face-detect/auto.sh >/home/pi/face-detect/cronlog 2>&1" >> autocron
crontab autocron
rm autocron