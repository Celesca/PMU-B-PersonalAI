# Mental Disorder Detection in Social Media

เราต้องการจะแยกคนที่มีภาวะซึมเศร้ากับอัตราเสี่ยงที่จะมีการฆ่าตัวตายบนโลกโซเชียลมีเดีย

![image](https://github.com/user-attachments/assets/42f6ef5b-3206-4bfd-beda-c8ae8ad9938a)

## 1. Data Collection

![image](https://github.com/user-attachments/assets/c050d9f1-6678-4a6a-b7d2-78a6a64ca07b)

![image](https://github.com/user-attachments/assets/12779e58-8b95-45bf-84a6-e898278a9b76)


* Forums - การพูดคุย Topics กัน
* Microblogs - ทำข้อความสั้นๆได้ เช่น Twitter
* Products/services review - ให้คนมารีวิวสินค้าว่ามันดีไม่ดี (เขาจะนำข้อมูลข้อความไปวิเคราะห์ประเภทลูกค้า)
* Social networks - มันจะให้คนสร้างโปรไฟล์ สามารถสนใจร่วมกันได้ โพสต์รูปภาพ โพสต์ลิ้งค์
* Photo sharing - Instagram แชร์รูปภาพ

ฉะนั้นวัตถุประสงค์หลักๆของแต่ละแพลตฟอร์ม เขาใช้ไปเพื่ออะไร เราจะศึกษาพฤติกรรมคนได้ถูกต้องของสื่อสังคม

การแบ่งประเภทของ Social Media ตามจุดมุ่งหมาย
![image](https://github.com/user-attachments/assets/0d72c5e2-ff57-4c5b-a15e-109ae4314795)

มันอยู่ที่วัตถุประสงค์ของคนว่าจะโพสตจ์ยังไง แล้วก็ Algorithm ว่าจะแลกเปลี่ยนความคิดกัน

![image](https://github.com/user-attachments/assets/22a64e53-9910-4ab3-8e84-3516494cb1b7)

User Data Collection

* สำหรับการเก็บข้อมูลเราก็จะเชิญเขามาร่วมกับงานวิจัยของเรา เช่น
  * สอบถามโดยตรงผ่าน Questionnaires ว่าเขามีภาวะซึมเศร้าอยู่หรือไม่ CSD เพื่อนำมา Label
  * EHR -> ได้รับการยืนยันจากแพทย์แล้วว่าเป็น
* Aggregating data extracted จากโพสต์ออนไลน์
  * I was diagnosed with [condition name]" เราก็จะมีนักวิจัยมา Annotate ว่าเขาพูดเกี่ยวกับมันจริงๆไหม เพื่อมา label
* Available Datasets

## 2. Data Exploration & Preprocessing

* Domain Knowledge
![image](https://github.com/user-attachments/assets/a17bd688-fb83-4894-a337-eb54d263e2d9)

พฤติกรรมซึมเศร้าเป็นยังไง

![image](https://github.com/user-attachments/assets/d59ca7bc-da3a-486c-b652-5a6015235221)

![image](https://github.com/user-attachments/assets/edcd5b30-2c30-4107-9e24-a7c8d1d2bf3f)

CountVectorizer -> ดู Bag of words ที่มี Frequency ของคำนั้นๆอยู่ โดยใช้ fit_transform (ดูว่ามีการใช้อยู่ในคอลัมน์ไหนบ้าง)

![image](https://github.com/user-attachments/assets/b6c13a78-b1f6-412a-9276-14a39fab8863)

LIWC-22 ในการใช้คำมากน้อยขนาดไหน เช่น เอาไว้ตรวจจับคำเชิงบวกหรือลบ

![image](https://github.com/user-attachments/assets/a99309e9-08b9-4de2-bd16-ee342538d543)

สีแดงคือคนที่มีภาวะซึมเศร้า กับคนที่มีความปกติ
เช่น ดูว่ามีการโพสต์บ่อยขนาดไหน ภาวะซึมเศร้าไม่ค่อยเคลื่อนไหว เขยื้อนตัวทำให้เขาโพสต์น้อย ไม่ค่อย Replies

การใช้คำเชิง Negative เชิงลบ แล้วก็ Activation ต่ำกว่าปกติ

อาจจะพูดถึงคนอื่นน้อย แล้วก็จะใช้คำว่า I (ตัวเองเยอะกว่า)
เกี่ยวกับคำสาบาน และ เกี่ยวกับภาวะเหนื่อย ซึมเศร้า Depression term คนซึมเศร้าจะมีมากกว่า

ดังนั้นเราจะสกัดข้อมูลออกมาได้อย่างถูกต้อง ตามพฤติกรรมของคนปกติ

![image](https://github.com/user-attachments/assets/ddff8db8-f9dc-4637-8ec6-97bbe5066d6d)

## 3. Predictive Modeling 

![image](https://github.com/user-attachments/assets/cb802f2e-7d9d-485e-ae58-3ffacc34278c)

![image](https://github.com/user-attachments/assets/68ec17a8-e21b-4d92-8e74-0eaa55f9215c)

เอาคำที่ได้มา negative / positive features ออกมา

![image](https://github.com/user-attachments/assets/28ae1162-05e4-4043-9130-67612c9b2d41)

## 4. Evaluation Model

![image](https://github.com/user-attachments/assets/e3176a0d-34f4-4f1c-90e6-1a13a92359ef)

ROC curves - ควรจะทำได้เหนือเส้นประ 

![image](https://github.com/user-attachments/assets/144a31c6-fb4a-4199-a432-be9ea9d46454)

## 5. Future Work

![image](https://github.com/user-attachments/assets/7aa6e377-9eeb-4c54-a6da-9f2dec84e7dd)

เช่นผ่านวิดีโอข้อความ ว่ามีโอกาสเป็นซึมเศร้าหรือไม่ซึมเศร้า คุณควรจะหาคนที่ให้คำปรึกษาดีไหม ทำให้คุณหมอสามารถให้คำปรึกษาได้ทัน
ฉะนั้นถ้าเขาไม่โพสต์ เราก็ไม่สามารถ Detect ได้ หากเราไม่สามารถ Detect จาก Media

