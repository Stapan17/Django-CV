from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
from .FaceDetector import FaceDetector

# Create your views here.


def handle_uploaded_file(f):
    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def home(request):
    return render(request, 'home.html')


def result(request):

    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])

        imageC = cv2.imread("img.jpg")
        gray = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)

        fd = FaceDetector("haarcascade_frontalface_default.xml")

        faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5,
                              minSize=(30, 30))

        for (x, y, w, h) in faceRects:
            cv2.rectangle(imageC, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite("./media/Face.jpg", imageC)

        context = {
            "numberOfFace": len(faceRects),
        }

        return render(request, 'result.html', context=context)

    return ""
