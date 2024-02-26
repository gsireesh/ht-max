import fitz
import json
import os

paperLocation = r'C:\Users\kevin\OneDrive\School\College\research\NLP\code\annotated_papers'
jsonLocation = r'C:\Users\kevin\OneDrive\School\College\research\NLP\code\AM_Creep_Papers_json'
# paperNames = [
#                 # "A creep-resistant additively manufactured Al-Ce-Ni-Mn alloy", 
#                 "Microstructural design of Ni-base alloys for high-temperature applications- impact of heat treatment on microstructure and mechanical properties after selective laser melting"
#               ]

highlightStruct = {1.0:'structure', 
                 0.15685999393463135:'property', 
                 0.0:'characterization', 
                 0.007843020372092724:'processing', 
                 0.2156829982995987:'materials', 
                 0.5254970192909241:'info'} 

def compilePaperData(filename, highlightStruct):
    pdf = fitz.open(filename)
    highlights = []
    highlightNumber = 0
    for i in range(len(pdf)):
        page = pdf[i]
        pageText = page.get_text_words()
        for annot in page.annots():
            boundingBoxes = []
            highlightText = ""
            if annot.type[1] == 'Highlight':
                highlightNumber += 1
                highlightData = dict()
                color = annot.colors['stroke']
                vertices = annot.vertices

                if len(vertices) == 4: #bounding box is 1 rect
                    boundingBoxes = [fitz.Quad(vertices).rect]
                else: #bounding box is multiple rect
                    for j in range(len(vertices)//4):
                        box = fitz.Quad(vertices[j*4:(j+1)*4]).rect
                        boundingBoxes.append(box)

                for box in boundingBoxes:
                    for word in pageText:
                        wordBox = fitz.Rect(word[0:4])
                        if box.intersects(wordBox): #checking if highlight contains words - maybe switch to letters?
                            highlightText += word[4] + ' '            

                highlightBoxes = [] # reformatting bboxes to JSON output
                for box in boundingBoxes:
                    #converting from fitz - [x0, y0, x1, y1] to papermage - [LEFT, TOP, WIDTH, HEIGHT, PAGE #]
                    highlightBoxes.append([box[0]/page.rect.width, box[1]/page.rect.height, (box[2]-box[0])/page.rect.width, (box[3]-box[1])/page.rect.height, i])

                boundingBoxes.append(i)
                highlightData["boxes"] = highlightBoxes
                highlightData["text"] = highlightText
                highlightData["category"] = highlightStruct[color[2]]
                highlightData["color"] = color
                highlights.append(highlightData)

    return highlights

def compileJSONData(filename):
    data = open(filename)
    jdata = json.load(data)
    jhighlights = jdata['entities']['tokens']
    return jhighlights

def compareJSONtoPaper(pData, jData):
    # Comparing overlap between JSON and Annotated Paper
        pLen = len(pData)
        jLen = len(jData)
        matches = []

        for h in range(pLen): #iterating through paper highlights
            pHighlight = pData[h]
            for j in range(jLen):
                jHighlight = jData[j]
                if pHighlight["boxes"][0][4] == pHighlight["boxes"][0][4]: #checking if highlights are on the same page.
                    for pBbox in pHighlight["boxes"]:
                        for jBbox in jHighlight["boxes"]:
                            pRect = fitz.Rect(pBbox[0:4])
                            jRect = fitz.Rect(jBbox[0:4])
                            if pRect.intersects(jRect):
                                matches.append(h,j)
                                print('Match')

def main():
    directory = os.fsencode(paperLocation+'\\')
    for filename in os.listdir(directory): #iterating through paper filenames in file
        filename = os.fsdecode(filename)
        if filename.endswith('.pdf'): # safety check 

            print(f'-------------{filename}')

            # Compiles highlight data from annotated paper
            pData = compilePaperData(paperLocation+'\\'+filename, highlightStruct) 
            print('Number of highlights = ', len(pData))
            
            # Outputs JSON file of annotated papers respective highlight data
            json_object = json.dumps(pData, indent = 4)
            with open(f'{filename[:-4]}.json', "w") as output:
                output.write(json_object)

            # Compiles JSON data from papermage reference
            # jData = compileJSONData(jsonLocation+'\\'+paperName+'.json') 
            # print('Number of JSON = ', len(jData))

            # compares input annotated papars to input JSON files to see common annoations
            # matches = compareJSONtoPaper(pData,jData)

main()
print('done')