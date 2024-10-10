from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import numpy as np
import io
import os
import shutil


import cv2
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import io
from google.cloud import vision





class GetInfo:
    def __init__(self, model_path="EDSR_x4.pb", subscription_key="", endpoint=""):

        self.model_path = model_path
        self.subscription_key = subscription_key
        self.endpoint = endpoint
    

    def upsample_image(self, image, output):
        image_data = cv2.imread(image)
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(self.model_path)
        sr.setModel("edsr", 2)
        self.upsampled_image = sr.upsample(image_data)
        cv2.imwrite(output, self.upsampled_image)

    def Azure_extract_text_lines(self, image):

        directory = '/Users/jung-yongjun/Desktop/details_translator/cropped_image'
        os.makedirs(directory, exist_ok=True)

        with open(image, 'rb') as f:
            image_data = f.read()

        line_coordinates = []  

        computervision_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))
        result = computervision_client.recognize_printed_text_in_stream(io.BytesIO(image_data))

        line_count = 0

        img = cv2.imread(image)

        for region in result.regions:
            for line in region.lines:
                bb = [int(num) for num in line.bounding_box.split(",")]
                x, y, w, h = bb[0], bb[1], bb[2], bb[3]
                cropped = img[y:y+h, x:x+w]

                save_path = os.path.join(directory, f'line_{line_count}.jpg')

                line_coordinates.append([x, y, w, h]) 
                cv2.imwrite(save_path, cropped)
                line_count += 1

        return line_coordinates 

    def Google_detect_text(self, url):
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

        client = vision.ImageAnnotatorClient()
        image = vision.Image()
        image.source.image_uri = url

        response = client.text_detection(image=image)
        texts = response.text_annotations

        main_text = texts[0].description
        main_text = main_text.split('\n')

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        
        return main_text


    def clear_folder(self):

        directory = '/Users/jung-yongjun/Desktop/details_translator/cropped_image'

        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Directory '{directory}' deleted.")
        else:
            print(f"Directory '{directory}' does not exist.")


    def Azure_detect_text(self, image):
        with open(image, 'rb') as image_file:
            image_data = image_file.read()

        computervision_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))
        result = computervision_client.recognize_printed_text_in_stream(io.BytesIO(image_data))

        main_text = []
        for region in result.regions:
            for line in region.lines:
                text = ''.join([word.text for word in line.words])
                main_text.append(text)

        return main_text



class Inpaint:
    def __init__(self):
        pass

    def process(self, image):
        info = GetInfo()

        img = cv2.imread(image)
        text_cordinate = info.Azure_extract_text_lines(image)

        num = len(text_cordinate)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for i in range(num):
            x = text_cordinate[i][0]
            y = text_cordinate[i][1]
            w = text_cordinate[i][2]
            h = text_cordinate[i][3]

            mask[y:y+h, x:x+w] = 255

            img[y:y+h, x:x+w] = 255

        directory = '/Users/jung-yongjun/Desktop/details_translator/cropped_image'
        save_path1 = os.path.join(directory, 'mask_image.jpg')
        save_path2 = os.path.join(directory, 'damaged_image.jpg')

        cv2.imwrite(save_path1, mask)
        cv2.imwrite(save_path2, img)


        #인페인팅
        recover = cv2.imread('/Users/jung-yongjun/Desktop/details_translator/cropped_image/damaged_image.jpg')
        recover_mask = cv2.imread('/Users/jung-yongjun/Desktop/details_translator/cropped_image/mask_image.jpg', cv2.IMREAD_GRAYSCALE)


        inpainted = cv2.inpaint(recover, recover_mask, 3, cv2.INPAINT_TELEA)

        save_path3 = os.path.join(directory, 'inpaint_img.png')
        cv2.imwrite(save_path3, inpainted)
    


class TranslateByGPT:
    def __init__(self):
        pass


    def process(self, text):
        import openai

        openai.organization = ""
        openai.api_key = ""

        response = openai.ChatCompletion.create(
            model ='gpt-3.5-turbo',
            messages=[
            {"role": "system", "content": f"너는 지금부터 구글 번역기보다 뛰어난 나의 전문 번역기야. 나에게 텍스트를 묻고 쇼핑몰 상품 상세페이지에 어울리는 자연스러운 어투와 문맥에 따라 이해하기 쉽게 의역을 해줘. 이해하지 못해도 적당히 번역을 해줘. 모든 텍스트는 한국어로 번역하고, 번역결과 이외의 다른 말은 일절 출력하지마"},
            {"role": "user", "content" : f"{text}"}
        ],
        )

        translation = response.choices[0].message["content"]
        translation = translation.encode('utf-8')
        translation = translation.decode('utf-8')

        return translation