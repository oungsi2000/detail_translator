from Tools import GetInfo, Inpaint, TranslateByGPT
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from PIL import Image
from ResNet_font_discriminator import MyModel
import subprocess


def predict_font(image_path):
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    num_font_types = 5
    num_font_weights = 5 ##여기를 실제 폰트 굵기의 개수로 바꾸기

    model = MyModel(num_font_types, num_font_weights)
    model.load_state_dict(torch.load("Font_Discriminator.pth"))
    model.eval() 


    img = Image.open(image_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)

    if torch.cuda.is_available():
        img = img.cuda()

    with torch.no_grad():
        out1, out2, out3 = model(img)

    predicted_font_type_idx = torch.argmax(out1, dim=1).item()
    predicted_font_weight_idx = torch.argmax(out2, dim=1).item()
    predicted_font_color = (torch.sigmoid(out3) * 255).squeeze().tolist()

    prediction = [predicted_font_type_idx, predicted_font_weight_idx, predicted_font_color]
    return prediction


#번역 이미지 생성 함수 만들기(크기, 폰트, 굵기, 색상) / 크롭된 각 이미지 개당 이미지 생성함 

def generate_translated_image(image_path, translate_idx=0):
    # machine_info = predict_font(image_path)

    with Image.open(image_path) as img:
        _, font_height = img.size

    # font_info = machine_info + font_height
    text = GetInfo().Google_detect_text(image_path)
    translated_text = TranslateByGPT().process(text[0])
    print(translated_text)

    #모델의 결과값에 따라 특정 폰트 사용하기 ㅇㅇ 굴기와 크기에 따라 다르게
    font_path = '/Users/jung-yongjun/websites/dalca4/among.software/wp-content/uploads/2021/06/SpoqaHanSansNeo-Regular.ttf'
    font_size = font_height
    font_color = (0,0,0)
    bg_color = (0,0,0,0)

    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = font.getsize(translated_text)

    image = Image.new('RGBA', (text_width, text_height), bg_color)
    d = ImageDraw.Draw(image)

    d.text((0, 0), translated_text, font=font, fill=font_color)
    
    directory = '/Users/jung-yongjun/Desktop/details_translator/cropped_image'
    save_path = os.path.join(directory, f'{translate_idx}_translated.png')
    image.save(save_path)
    



#이미지 붙여넣기 함수 제작

def paste(bg_image_path, image_path, coordinate_image_path, coordinate_idx=0):
    background_img = Image.open(bg_image_path).convert("RGBA")
    target_img = Image.open(image_path).convert("RGBA")

    coordinates = GetInfo().Azure_extract_text_lines(coordinate_image_path)
    x, y, w, h = coordinates[coordinate_idx]
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the upper left coordinate for pasting
    target_width, target_height = target_img.size
    paste_x = center_x - target_width // 2
    paste_y = center_y - target_height // 2

    # Make sure the image is not out of bounds
    paste_x = max(paste_x, 0)
    paste_y = max(paste_y, 0)

    coordinate = (paste_x, paste_y)
    merged_img = background_img.copy()

    # Paste the target image onto the merged image using alpha compositing
    merged_img.alpha_composite(target_img, coordinate)
    merged_img.save(bg_image_path)



image_path = 'chinese.jpeg'
Inpaint().process(image_path)
GetInfo().Azure_extract_text_lines(image_path)


#각 크롭된 텍스트 번역
directory = '/Users/jung-yongjun/Desktop/details_translator/cropped_image'
start_char = 'line'
num_image = os.listdir(directory)
num_image = [f for f in num_image if f.startswith(start_char)]

j = 0
for i in num_image:
    generate_translated_image(f'{directory}/{i}', translate_idx=j)
    img = Image.open('chinese.jpeg')
    rgba_img = img.convert('RGBA')
    rgba_img.save('chinese.png', 'PNG')

    paste(f'{directory}/inpaint_img.png', f'{directory}/{j}_translated.png', 'chinese.png', coordinate_idx=j)
    j += 1


old = f'{directory}/inpaint_img.png'
new = 'completed_image.jpg'
os.rename(old, new)

GetInfo().clear_folder()

print('done')

###현재 상페 번역기의 문제점
# 1. chatgpt 의 번역 문제 
# 현재 챗지피티의 번역에는 쓸데없는 미사여구 또는 번역에 실패하는 경우가 있음
# 2. 번역된 텍스트 크기, 좌표 문제
# 현재 번역된 텍스트를 인페인트된 이미지에 붙여넣으면 크기가 안맞아서 디자인적으로 더러워짐
# 3. 문단 단위 번역 문제
# 현재는 문맥에 상관없이 물리적으로 단나눔이 되어 있으면 각 문장을 별도의 문장으로 해석함, 번역의 정확도가 떨어짐
# 4. 배경에 따른 폰트 컬러 적용 문제


### 해결 방법
#1. chatgpt 프롬프트 연구 및 조교, 안될 시, 다른 번역 프로그램으로 전환하면 해결됩니다
#2. 번역된 텍스트 크기는 최대한 원본 길이에 맞게 번역하게끔 조교 (챗지피티 이용 시), 원본 텍스트 이미지 크기와 일치하게 폰트 크기를 키우거나 줄여서 좌표에 딱 맞게 만든다
#3. 일단 각 텍스트를 번역하기 전, 구글 텍스트 감지로 인식된 텍스트 전부를 번역한다(이렇게 번역하면 문맥에 맞게끔 번역이 가능할 테니) -> 그 다음 줄나눔 별로 다시 문장을 나눈다
#4. 배경 부분을 최대한 지우고 (cv2 이용)numpy의 kmean 연산으로 이미지에서 차지하는 주된 픽셀을 알아내서 폰트 색상 정하기

###이거보단 그냥 셀레니움에서 구글, 파파고 이미지 번역 크롤링을 하는게 더 싸고 더 정확하고 더 빠름 ㅋㅋ
###일단 상페 번역은 나의 수준에 맞지 않았다는 것을 인정하고, 네이버, 지옥션 상품 업로드로 넘어가자.