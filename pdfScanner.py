import fitz
import tabula # pip install tabula-py[jpype] - 
#paperMage - spits back image highlighted of table
# save bounding does -> segway into papermage objects

class paper:
    def __init__(self, filename):
        self.filename = filename
        self.highlightDict = {'structure':[], 'property':[], 'processing':[], 'characterization':[], 'materials':[], 'info':[]}
        self.highlightNumber = 0
        self.tables = []
        self.annotated = False
        self.tabulated = False

    def __repr__(self):
        return self.printHighlights()

    def compileHighlights(self, location, highlightStruct):
        if self.annotated == True:
            return 'Already Annotated'
        pdf = fitz.open(location+self.filename)
        for i in range(len(pdf)):
            page = pdf[i]
            text = page.get_text_words()
            for annot in page.annots():
                boundingBoxes = []
                annotation = ""
                if annot.type[1] == 'Highlight':
                    self.highlightNumber += 1
                    vertices = annot.vertices
                    if len(vertices) == 4: #bounding box is 1 rect
                        annotation = ""
                        boundingBoxes = [fitz.Quad(vertices).rect]
                    else: #bounding box is multiple rect
                        for j in range(len(vertices)//4):
                            box = fitz.Quad(vertices[j*4:(j+1)*4]).rect
                            boundingBoxes.append(box)
                    for box in boundingBoxes:
                        for word in text:
                            wordBox = fitz.Rect(word[0:4])
                            if box.contains(wordBox): #checking if highlight contains words 
                                annotation += word[4] + ' '
                    self.highlightDict[highlightStruct[annot.colors['stroke'][2]]].append(annotation) #adding annotation to dictionary based on highlight color
        self.annotated = True
        return self.highlightDict
    
    def printHighlights(self):
        data = ''
        for key in self.highlightDict:
            data += f'\n-----{key}-----\n\n'
            for highlight in self.highlightDict[key]:
                data += f'-{highlight}\n'
        return data
    
    def compileTables(self, location):
        if self.tabulated == True:
            return 'Already Tabulated'
        self.tables = tabula.read_pdf(location+self.filename, pages = "all")
        self.tabulated = True
        for table in self.tables:
            print(table)
            print('-----------------------------')
 
        tabula.convert_into("test.pdf", "output.json", output_format="json", pages='all')


location = "C:\\Users\\kevin\\OneDrive\\School\\College\\research\\Paper Annotations\\Annotated\\"
pdfs = ["annotated_Creep deformation and failure properties of 316 L stainless steel manufactured by laser powder bed fusion under multiaxial loading conditions.pdf",]

highlightStruct = {1.0:'structure', 
                 0.15685999393463135:'property', 
                 0.0:'characterization', 
                 0.007843020372092724:'processing', 
                 0.2156829982995987:'materials', 
                 0.5254970192909241:'info'} 

def populatePapers(files, location, highlightStruct):
    papers = []
    for file in files:
        pdf = paper(file)
        pdf.compileHighlights(location, highlightStruct)
        #pdf.compileTables(location)
        print(pdf)
        papers.append(pdf)

def main():
    papers = populatePapers(pdfs, location, highlightStruct)

main()

#(blue, green, yellow, orange, red, purple ) keys are based on B value of RGB tuple

# pdf = fitz.open(location+paper)
# highlightNumber=0
# for i in range(len(pdf)):
#     page = pdf[i]
#     text = page.get_text_words()
#     for annot in page.annots():
#         boundingBoxes = []
#         annotation = ""
#         if annot.type[1] == 'Highlight':
#             highlightNumber += 1
#             vertices = annot.vertices
#             if len(vertices) == 4: #bounding box is 1 rect
#                 annotation = ""
#                 boundingBoxes = [fitz.Quad(vertices).rect]
#             else:                  #bounding box is multiple rect
#                 highlights = []
#                 for j in range(len(vertices)//4):
#                     boundingBox = fitz.Quad(vertices[j*4:(j+1)*4]).rect
#                     boundingBoxes.append(boundingBox)

#             for boundingBox in boundingBoxes:
#                 for word in text:
#                     wordBox = fitz.Rect(word[0:4])
#                     if boundingBox.contains(wordBox): #checking if highlight contains words 
#                         annotation += word[4] + ' '
#             annotationDict[highlightType[annot.colors['stroke'][2]]].append(annotation) #adding annotation to dictionary based on highlight color





