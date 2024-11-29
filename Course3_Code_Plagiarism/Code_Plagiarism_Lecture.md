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


## 3. การนำ Machine Learning Model มาต่อกัน

![image](https://github.com/user-attachments/assets/ea292c0d-7e0a-4316-b128-c9f2d6804421)

เอา Methods ไปผ่าน Code2Vec ก็จะได้ Similarity เป็น Synthetic Code Metrics

Java Tokenizer (ANTLR) เป็นก้อนๆ มาแตก Variables, Line of code, methods จะได้เป็น Semantic Code Metrics

จากนั้นเอาทั้งสองมาผ่าน WEKA (Trained ML Model) เอาไฟล์ที่ได้มา ผ่านตัวเทรนด์ว่าจะเป็นคู่บ้าง

![image](https://github.com/user-attachments/assets/dab3b74c-243e-44a6-850f-989d1d07f8a5)

มีซอสโค้ดเหมือนกันมากแค่ไหนของนักศึกษา

![image](https://github.com/user-attachments/assets/e4dc23e3-979c-4407-b58c-c6690afc46a7)

![image](https://github.com/user-attachments/assets/9e002eff-db15-41cb-97dc-d1ebaeccc4eb)

## 4. การวัดประสิทธิภาพของ ML Model ที่สร้างขึ้น ในเชิงความแม่นยำ

4.1 How accurate is Merry code clone ใน data BigCloneBench (BCB) มันมีความแม่นยำขนาดไหน

![image](https://github.com/user-attachments/assets/e8b2ffdf-bf88-445e-a118-8d72ffe8707c)

Precision -> ความแม่นยำ
Recall -> ข้อมูลที่เรารู้ว่าถูก เราดึงมาได้เท่าไหร่ (เจอโค้ดที่เหมือนกันมากขึ้น)
F1-Score

![image](https://github.com/user-attachments/assets/48088afe-21eb-4f4e-aa68-66fc1871a46b)

เราหยิบข้อมูล Test Data มาผ่าน Clone metric extractor 11 + 12 metrics จับคู่ แล้วผ่าน Trained ML Model ว่ามันใช่จริงไหม

![image](https://github.com/user-attachments/assets/9a125d3b-4e18-402a-af5a-ce7c88b6390c)

Baseline เป็น อันนี้ใช่และไม่ใช่ โดยการ Randomization
มันบอกให้เห็นว่า เราต้องใช้สองอันมารวมกัน มันสูงที่สุด ดูโครงสร้างอาจจะไม่เพียงพอ ต้องใช้การดูฟังก์ชัน (Synthetic) ด้วย

4.2 ถ้าเอา Merry code ไปรันโปรเจ็คจริงๆ on Real Software Projects

![image](https://github.com/user-attachments/assets/e25132f6-0bb4-4b3f-ba7a-ab3a6275cf9f)

เราเอาผู้เชี่ยวชาญมาช่วยตรวจข้อสอบตรงนี้

![image](https://github.com/user-attachments/assets/bf40204d-d58c-45ca-ac31-0295b0c2d966)

ตัวอย่าง

![image](https://github.com/user-attachments/assets/e6202e9c-071f-4e5f-a81f-3d559a4c28de)


4.3 มีความง่ายในการใช้งานมากแค่ไหน










