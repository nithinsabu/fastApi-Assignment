import os
import json
from ultralytics import YOLO
import urllib.request
categories = {1: "person", 3: "car", 10: "traffic light", 17: "cat", 27: "backpack"}
yolo_categories = {1:0, 3:2, 10:9, 17:15, 27:24}
def generate_data():
    with open(os.path.join(os.path.dirname(__file__), "instances_val2017.json")) as f:
        data = json.load(f)

    images = data["images"]
    for i in images:
        i["object_classes"] = []

    annotations = data["annotations"]
    test_data = {i: {"positive": [], "negative": []} for i in categories}
    for i in annotations:
        image = next(image for image in images if image["id"]==i["image_id"])
        if i["category_id"] not in image["object_classes"]:
            image["object_classes"].append(i["category_id"])

    for i in images:
        objects = i["object_classes"]
        for j in objects:
            if j in test_data and len(test_data[j]["positive"])<40:
                test_data[j]["positive"].append(i["coco_url"])
                break
        else:
            for j in test_data:
                if len(test_data[j]["negative"])<40:
                    test_data[j]["negative"].append(i["coco_url"])
                    break
    with open(os.path.join(os.path.dirname(__file__), "images.json"), "w") as f:
        json.dump(test_data, f)
# generate_data()

def test_model(save:bool = True):
    model = YOLO("yolo11n.pt")
    test_images_dir = os.path.join(os.path.dirname(__file__), "val2017")
    if not os.path.exists(test_images_dir):
        os.mkdir(test_images_dir)
    existing_images = os.listdir(test_images_dir)
    with open(os.path.join(os.path.dirname(__file__), "images.json"), "r") as f:
        test_data = json.load(f)
    
    #download missing images
    for i in test_data:
        for image in test_data[i]["positive"]+test_data[i]["negative"]:
            image_name = image.split('/')[-1]
            if image_name not in existing_images:
                print("Downloaded", image_name)
                urllib.request.urlretrieve(image, os.path.join(test_images_dir, image_name))

    #dictionary of {YOLO class: {positive: [file paths], negative: [file paths]}}
    test_data_file_path = {i: {"positive": [], "negative": []} for i in yolo_categories.values()}
    for i in test_data:
        test_data_file_path[yolo_categories[int(i)]]["positive"] = [os.path.join(test_images_dir, filename.split('/')[-1]) for filename in test_data[i]["positive"]]
        test_data_file_path[yolo_categories[int(i)]]["negative"] = [os.path.join(test_images_dir, filename.split('/')[-1]) for filename in test_data[i]["negative"]]
    
    #testing
    # test_results = {cls: {} for cls in test_data_file_path}
    TP = FN = TN = FP = 0
    for cls in test_data_file_path:
        results_for_positive = model.predict(test_data_file_path[cls]["positive"])
        for result in results_for_positive:
            object_classes = list(result.boxes.cls)
            if cls in object_classes:
                TP+=1
            else:
                FN+=1
        # print(f"Class {cls}: TP={TP} FN={FN}")

        results_for_negative = model.predict(test_data_file_path[cls]["negative"])
        for result in results_for_negative:
            object_classes = list(result.boxes.cls)
            if cls in object_classes:
                FP+=1
            else:
                TN+=1
        # print(f"Class {cls}: FP={FP} TN={TN}")
        # test_results[cls] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    f1_score = 2/(1/recall+1/precision)
    return (recall, precision, accuracy, f1_score)

recall, precision, accuracy, f1_score = test_model()
print (f"Recall: {recall}, Precision: {precision}, Accuracy: {accuracy}, F1 Score: {f1_score}")

        
