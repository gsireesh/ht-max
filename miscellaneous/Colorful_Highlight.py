import fitz  #PyMuPDF

'''
structure (eg.microstructure/phase/macrostructure): blue

property (eg.mechanical property/creep resistance):green 

characterization (eg. EBSD/SEM/TEM): yellow

processing (eg.machine/heat treatment/scanning speed/power/other parameters in AM): orange
materials: red
other important information: purple
'''

Blue = [0.0, 0.0, 1.0]
Green = [0.4156799912452698, 0.8509830236434937, 0.15685999393463135]
Yellow = [1.0, 0.8196110129356384, 0.0]
Orange = [1.0, 0.4392090141773224, 0.007843020372092724]
Red = [0.8980410099029541, 0.13333100080490112, 0.2156829982995987]
Purple = [0.6392210125923157, 0.18823200464248657, 0.5254970192909241]

Color_Class_Map = { #Last channel (B) seems unique so it can be used for hashing
    Blue[-1]: ["blue", "Structure"],
    Green[-1]: ["green", "Property"],
    Yellow[-1]: ["yellow", "Characterization"],
    Orange[-1]: ["orange", "Processing"],
    Red[-1]: ["Red", "Materials"],
    Purple[-1]: ["purple","Other Important Information"]
}

#'/Users/harryzhang/Downloads/testPDF.pdf'

def get_colorful_highlights(PDF_Path):
    # Open the PDF
    doc = fitz.open(PDF_Path)

    # Category - Highlights Mapping
    colorful_highlights = {
        "Structure":[],
        "Property":[],
        "Characterization":[],
        "Processing":[],
        "Materials":[],
        "Other Important Information":[]
    }

    # Loop through every page
    for i in range(len(doc)):
        page = doc[i]
        # Get the annotations (highlights are a type of annotation)
        annotations = page.annots()

        for annotation in annotations:
            if annotation.type[1] == 'Highlight':
                # Get the color of the highlight
                color = annotation.colors['stroke']  # Returns a RGB tuple

                color_B = color[-1]

                color_name, category = Color_Class_Map[color_B]
                
                colorful_highlights[category].append(annotation.rect)

    return colorful_highlights
            

if __name__ == "__main__":

    Highlights = get_colorful_highlights('/Users/harryzhang/Downloads/testPDF.pdf')

    print(Highlights)




