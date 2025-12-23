import cv2
import numpy as np
import onnxruntime as ort
import os
import sys

MODEL_PATH="models/yolov8n.onnx"
IMAGE_PATH="test.jpg"
CONF_THRESHOLD=0.5
NMS_THRESHOLD=0.45
PERSON_CLASS_ID=0

if not os.path.exists(MODEL_PATH):
    print("Model not found")
    sys.exit(1)

if not os.path.exists(IMAGE_PATH):
    print("Image not found")
    sys.exit(1)

session=ort.InferenceSession(MODEL_PATH,providers=["CPUExecutionProvider"])
input_meta=session.get_inputs()[0]
input_name=input_meta.name
in_shape=input_meta.shape

if len(in_shape)==4:
    model_h=in_shape[2] if in_shape[2] else 640
    model_w=in_shape[3] if in_shape[3] else 640
else:
    model_h=model_w=640

img=cv2.imread(IMAGE_PATH)
h_orig,w_orig=img.shape[:2]

img_resized=cv2.resize(img,(model_w,model_h))
img_rgb=cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
img_norm=img_rgb.astype(np.float32)/255.0
img_chw=np.transpose(img_norm,(2,0,1))
input_tensor=np.expand_dims(img_chw,axis=0)

outputs=session.run(None,{input_name:input_tensor})
pred=outputs[0]

if pred.ndim==3 and pred.shape[1]==84:
    preds=np.transpose(pred,(0,2,1))[0]
elif pred.ndim==3 and pred.shape[2]==84:
    preds=pred[0]
else:
    preds=pred

boxes=[]
confidences=[]

x_factor=w_orig/model_w
y_factor=h_orig/model_h

for row in preds:
    bbox=row[0:4]
    class_scores=row[4:]
    max_score=float(np.max(class_scores))
    class_id=int(np.argmax(class_scores))

    if max_score<CONF_THRESHOLD:
        continue
    if class_id!=PERSON_CLASS_ID:
        continue

    cx,cy,bw,bh=bbox
    left=int((cx-bw/2)*x_factor)
    top=int((cy-bh/2)*y_factor)
    width=int(bw*x_factor)
    height=int(bh*y_factor)

    boxes.append([left,top,width,height])
    confidences.append(max_score)

indices=cv2.dnn.NMSBoxes(boxes,confidences,CONF_THRESHOLD,NMS_THRESHOLD)

print("Detected persons:",len(indices))

for i in indices.flatten():
    x,y,w_box,h_box=boxes[i]
    conf=confidences[i]
    cv2.rectangle(img,(x,y),(x+w_box,y+h_box),(0,255,0),2)
    cv2.putText(img,f"Person {conf:.2f}",(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

cv2.imshow("Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
