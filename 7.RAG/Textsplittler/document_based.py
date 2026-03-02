from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

code = """
class OCRdataset(Dataset):
   def __init__(self, img_paths, words, transform=None):
        self.img_paths = img_paths
        self.words     = words
        self.transform = transform
   def __len__(self):
        return len(self.words)
   def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        words = [SOS_TOKEN] + encode(self.words[idx]) + [EOS_TOKEN]
        words = torch.tensor(words, dtype=torch.long)
        return image, words
        train_dataset = OCRdataset(img_path, words, Transform)
train_Loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)"""

splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=200, chunk_overlap=0, language=Language.PYTHON)

chunks = splitter.split_text(code)

for i, chunk in enumerate(chunks):  
    print(f"\nChunk {i+1}:")
    print(chunk)
