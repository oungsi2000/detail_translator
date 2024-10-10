import os
import random
import string
import unicodedata
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from fontTools.ttLib import TTFont
import json



# font_path = ['/Users/jung-yongjun/Downloads/CN-고딕','/Users/jung-yongjun/Downloads/CN-명조','/Users/jung-yongjun/Downloads/CN-장식체','/Users/jung-yongjun/Downloads/CN-고전체']

# def random_font(type):

#     fonts = []
#     for i in font_path:
#         with open(f'{i}.txt', 'r') as file:
#             fonts.append([line.strip() for line in file.readlines() if line.strip().lower().endswith(('.ttf', '.otf'))])
#     font = random.choice(fonts[type])
#     return font


# def generate_random_chinese(length, type):
#     chinese_string = ""
#     while len(chinese_string) < length:
#         while True:
#             codepoint = random.randint(0x4e00, 0x9fff)
#             character = chr(codepoint)

#             if unicodedata.name(character).startswith('CJK UNIFIED IDEOGRAPH'):
#                 try:
#                     # 폰트에 해당 문자가 지원되는지 확인
#                     font_path = random_font(type)
#                     font = ImageFont.truetype(font_path, 12)  # 원하는 폰트 크기로 설정
#                     font.getmask(character)
#                     chinese_string += character
#                     break
#                 except Exception:
#                     continue
#     return chinese_string


    #수정해야 할 것
    #1. 폰트 엑스박스 뜨는 문제 해결(이건 거의 다 됨) -> 해결
    #2. 폰트 경로 모두 .ttf 아니면 .otf 만 있게 설정  -> 해결
    #3. resnet 모델 수정
    #4. 폰트 재다운로드

    #문제 1. 텍스트를 만들 때 글자 개수를 제한하면 될려나??


class ImageCreator:
    def __init__(self):
        self.data = {}
        self.font_types = []  # 폰트 종류를 저장하는 리스트
        self.font_weights = []  # 폰트 굵기를 저장하는 리스트
        try:
            with open('labels.json', 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            pass

    def get_all_paths(self, directory):
        paths = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                paths.append(os.path.join(root, name))

        with open(f'{directory}.txt', 'w') as f:
            for path in paths:
                f.write(path + '\n')

    def check_font_support(self, font_path, text):
        font = TTFont(font_path)
        support_all = True
        for character in text:
            support_character = False
            for table in font['cmap'].tables:
                if ord(character) in table.cmap:
                    support_character = True
                    break
            if not support_character:
                support_all = False
                break
        return support_all


    #[0]:고딕 [1]:명조 [2]:장식체 [3]:고전체 [4]:손글씨
    def random_font(self, type):

        #저장된 폰트 경로 txt 파일
        font_path = ['/Users/jung-yongjun/Downloads/CN-고딕',
        '/Users/jung-yongjun/Downloads/CN-명조',
        '/Users/jung-yongjun/Downloads/CN-장식체',
        '/Users/jung-yongjun/Downloads/CN-고전체',
        '/Users/jung-yongjun/Downloads/CN-손글씨'
        ]


        fonts = []
        for i in font_path:
            with open(f'{i}.txt', 'r') as file:
                fonts.append([line.strip() for line in file.readlines() if line.strip().lower().endswith(('.ttf', '.otf'))])
        font = random.choice(fonts[type])
        return font


    #length: 문자 길이 #type 폰트 종류, 자세한 것은 아래 함수 참조
    def generate_random_chinese(self, length, type):
        chinese_string = ""
        while len(chinese_string) < length:
            while True:
                codepoint = random.randint(0x4e00, 0x9fff)
                character = chr(codepoint)

                if unicodedata.name(character).startswith('CJK UNIFIED IDEOGRAPH'):
                    try:
                        # 폰트에 해당 문자가 지원되는지 확인
                        font_path = self.random_font(type)
                        font = ImageFont.truetype(font_path, 12)  # 원하는 폰트 크기로 설정
                        font.getmask(character)
                        chinese_string += character
                        break
                    except Exception:
                        continue
        return chinese_string


    def create_image(self, idx, kind, kind_idx, color_bg='random', color_text='random'):
        # 함수에서 공통적으로 사용되는 부분을 선언
        random_length = random.randint(1,10)
        text = self.generate_random_chinese(random_length, 0)
        font_path = self.random_font(kind_idx)  # 폰트 파일의 경로
        font_size = random.randint(12,100)  # 폰트 사이즈
        font_support = self.check_font_support(font_path, text)
        
        while not font_support:
            random_length = random.randint(1,10)
            text = self.generate_random_chinese(random_length, 0)
            font_path = self.random_font(kind_idx)  # 폰트 파일의 경로
            font_size = random.randint(12,100)  # 폰트 사이즈
            font_support = self.check_font_support(font_path, text)

        font = ImageFont.truetype(font_path, font_size)
        w = TTFont(font_path)
        font_weight = w['OS/2'].usWeightClass

        text_width, text_height = font.getsize(text)

        # 색상 설정
        if color_bg == 'random':
            bg_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        elif color_bg == 'black':
            bg_color = (0, 0, 0)
        elif color_bg == 'white':
            bg_color = (255, 255, 255)

        if color_text == 'random':
            text_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        elif color_text == 'black':
            text_color = (0, 0, 0)
        elif color_text == 'white':
            text_color = (255, 255, 255)

        image = Image.new('RGB', (text_width, text_height), color=bg_color)
        d = ImageDraw.Draw(image)
        d.text((0,0), text, fill=text_color, font=font)

        image_name = f'text_image_{idx}_{kind_idx}_{color_bg}_{color_text}.png'
        self.font_types.append(kind)
        self.font_weights.append(font_weight)

        save_path = '/Users/jung-yongjun/Desktop/details_translator/images'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image.save(os.path.join(save_path, image_name))

        self.data[image_name] = {
            '폰트종류': kind,
            '폰트굵기': font_weight,
            '폰트색상': f'{text_color[0]},{text_color[1]},{text_color[2]}'
        }

        with open('labels.json', 'w', encoding='UTF-8') as file:
            json.dump(self.data, file, ensure_ascii=False)

for i in range(30):
    ImageCreator().create_image(i, '고딕', 0, color_bg='black', color_text='white')

    