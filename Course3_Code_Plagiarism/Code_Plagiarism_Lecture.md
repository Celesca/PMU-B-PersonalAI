# AI for Code Clone Detection (MERRY)

## 1. What is Code Clones

![image](https://github.com/user-attachments/assets/104a40f3-86c7-4eb8-af17-b2a6bea474af)

แบ่งได้หลายชนิด 
* Type 1 : มีความเหมือนกันแบบเป๊ะๆ 
* Type 2 : เหมือนกันต่างกันแค่ตัวแปร
* Type 3 : อาจมีบางบรรทัดแก้ไข มีการตรวจสอบยาก เช่น return a + b; Refactoring เหลือบรรทัดเดียว
* Type 4 : ชนิดที่เพียงมีการคำนวณเหมือนกัน แต่ฟังก์ชันไม่เหมือนกัน (2 Method ไม่เหมือนกัน) ตรวจสอบยากที่สุด

![image](https://github.com/user-attachments/assets/b3a51efd-4de6-4475-b7b2-bd4f63f83d25)

ถ้ามี Source Code มันมีปัญหามากๆ มันจะ Maintain ได้ยาก ถ้าหากมันเกิด Bug

![image](https://github.com/user-attachments/assets/5cb1a17c-9071-44ee-8e91-188cf4f5e00c)

ปัญหาที่เจอในปัจจุบันก็คือ

1. เครื่องมือที่มีอยู่ใช้ยากต่อการตรวจจับ Several modifications
2. เครื่องมือ Clone detection กับ plagiarism detection ใช้ยากเพราะเป็น Command-line based tool

Objectives :

1. สร้างเครื่องมือตรวจ Code Clone Detection โดยใช้ Machine learning and study its effectiveness
2. เพื่อเพิ่ม user Experience ให้กับ Code Clone Detection Tools โดยทำเป็น เว็บแอพ กับ Visualization Clone Results

![image](https://github.com/user-attachments/assets/7ea27cf5-ceb2-46a0-b3ea-86ad4cf9dc67)

## 2. วิธีการสร้าง Machine Learning Model

1. Building Merry Engine

  1.1 Data collection and preparation

![image](https://github.com/user-attachments/assets/b652a6b5-191f-40b1-bd8e-dc75b653a309)

Label ว่ามัน Clone จริงหรือไม่จริง

![image](https://github.com/user-attachments/assets/10152d9a-4f73-4cb7-9051-6176cfde57d2)

ได้ชื่อไฟล์ บรรทัดเริ่มต้น บรรทัดจบ จากนั้นเอาไปเสิร์จ Source Code จริงๆ เป็นส่วนหนึ่งของไฟล์ จากนั้นเอาข้อมูลไปเสิร์จจริงๆ
ทำที่เหมือนกันและไม่เหมือนกัน เราก็จะแบ่ง Training Set and Testing Set Splitting

![image](https://github.com/user-attachments/assets/24e26621-35e3-455d-bf38-a07e3ee06e3f)

ให้มันมีจำนวนเท่ากันใน Training Data

  1.2 Code metrics extraction

Syntactic Code metrics ดูจาก Syntax เป็นกฎที่ต้องเขียนอย่างไร หาจากโครงสร้างของโค้ด

เราจะเอาค่า Metrics ออกมา เช่น จำนวนโค้ด จำนวน Token (int void main จำนวนตัวแปร) เอามาลบกัน

![image](https://github.com/user-attachments/assets/ea2ea3bb-0c4c-4f6d-a35e-010e29897de5)

![image](https://github.com/user-attachments/assets/8ba00d20-f312-4993-a689-8472c5c025b7)

Semantic Code metrics ไม่ได้ดูแค่ตัวโครงสร้างโค้ด แต่ดูการทำงาน พยายามเข้าใจฟังก์ชันโค้ด

เราใช้ Pretrained-model (Code2Vec) มันจะสร้างเวกเตอร์ออกมา มันจะมีความเหมือนกันของเวกเตอร์ เช่น Cosine Similarity 

* Code2Vec in Detail

![image](https://github.com/user-attachments/assets/73a2a508-9137-4a0a-ba32-9bdb93475fcf)

เราสร้าง Source Code -> Tree โดยที่มี Method จะถูกแปลงเป็น Syntax Speech แล้วจะดึงของแต่ละก้อนของ Subtree ออกมาเป็น Context Vectors
ผ่าน Fully-connected layer แล้วใส่ Attention weights

![image](https://github.com/user-attachments/assets/3011e899-b3d7-4810-9811-8475470ff2ad)

  1.3 Machine Learning Model

![image](https://github.com/user-attachments/assets/eba01a9c-3d63-4fec-90b7-db4749f62bad)

เราใช้ SVM using SMO Optimization ศึกษาผ่านของ WEKA








