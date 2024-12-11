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


