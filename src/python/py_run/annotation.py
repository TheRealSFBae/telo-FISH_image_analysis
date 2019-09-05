import xlrd, os
from statistics import median

# Telo is count of number of nuclei reviewed
# Median is median of [telomere intensity / nuclear surface area] for those nuclei

start_dir = "/Users/apelonero/git/Telofish_NN/summaries/"
output_filename = "annotations_ang.csv"
with open(output_filename, "wb") as output_file:
    output_file.write("File,Median,Telo\n")

    for filename in os.listdir(start_dir):
        try:
            wb = xlrd.open_workbook(start_dir+filename)
            sheet = wb.sheet_by_index(0)
            
            #print(filename, sheet.nrows)
            telo = []
            image = sheet.cell_value(0,0)
            for row in xrange(sheet.nrows):
                if (image != sheet.cell_value(row,0)):
                    output_file.write(", ".join([image.split("\\")[-1], str(median(telo)), str(len(telo))]) + "\n")
                    telo = []
                    image = sheet.cell_value(row,0)
                telo.append(sheet.cell_value(row, 2))
            #print telo
        except xlrd.XLRDError:
            print ("error:", filename)
            next
        except AttributeError:
            print "Attribute error:", filename
            next
        except TypeError:
            print "Unicode crap again:", filename
            next


            
