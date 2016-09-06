#german_parser
#Li Zhao
#09/04/2016

#A parser that gets the sentence level data from SALSA.

# Comments outline the XML tree
#children of root (tag, attributes)
#('head', {})
#('body', {})

#children of head (tag, attributes)
#('meta', {})
#('annotation', {})

# print "head:"
# for child in root[0]:
# 	print(child.tag, child.attrib)

#children of body (tag, attributes)
#.....up to
#('s', {'id': 's50473'})

#look at the first one as example
#('graph', {'root': 's2_VROOT'})
#('matches', {})
#('sem', {})


# graph:
# ('terminals', {})
# ('nonterminals', {})	
# matches:
# sem:
# ('globals', {})	
# ('frames', {})	
# ('usp', {})  
# ('wordtags', {})

# globals
# ()

# frames
# (frame)
# frame
# (target, fe)

# usp
# (uspframes, uspfes)
# wordtags
# ()

import xml.etree.ElementTree as ET
import shutil
import os

corpora_path = "./salsa-corpora/"
parsed_path = "./salsa-parsed/"

def parse_file(name):
	#open file
	tree = ET.parse(corpora_path+ name)
	root = tree.getroot()
	sentences = root.findall(".//s")

	#make parsed file
	nameonly = os.path.splitext(name)[0]
	#datafile = open(nameonly+".txt", "w")
	datafile = open(parsed_path +nameonly+".txt", "w")

	# counter for samples
	used = 0
	unused = 0

	#for each sentence
	for sentence in sentences:

	#get the sentence as a list
		nodelist = sentence.findall(".//t")

	#get the list of non terminals to check, incase any of our words are not words but NTs
		nonterminals = sentence.findall(".//nt")

	#get the related words
		relatedwords = sentence.findall(".//frame//fe/fenode")

		finalrelatedwords = []
		for each in relatedwords:
			label = each.attrib["idref"]
			relatedwords[relatedwords.index(each)] = label
		#print (relatedwords)
		for part in relatedwords:
			if (len(sentence.findall(".//nt/[@id='part']"))>0):
				relatedid = list(sentence.findall(".//nt/[@id='part']")[0])
				for partition in relatedid:
					newlabel = partition.attrib["idref"]
					finalrelatedwords.append(newlabel)
			else:
				finalrelatedwords.append(part)
		#print (finalrelatedwords)
		#get the target word, if multiple, then skip this sentence, increase the counter



		if (sentence.findall(".//frame/target") != []):
				targetwords = list(sentence.findall(".//frame/target")[0])
				if (len(targetwords)==1 and (sentence.findall(".//frames/frame")!=[])):

					#get the frame
					frameid = sentence.findall(".//frames/frame")[0].attrib["id"]
					#some do not have frames and have to be skipped
					
					#make the string
					#format:
					#Frame_label<tab>word1 word2keywordtag word3<tab>target_verb<tab>word4 word5
					result = frameid + "\t"

					# finalresult = []
					# for i in range((len(nodelist)*2)+2):
					# 	finalresult.append("")
					# finalresult[0] = frameid
					# finalresult[1] = "\t"

					#print (nodelist)
					for node in nodelist:
						if (node.attrib['id'] in finalrelatedwords):
						#this is a related word
							result = result + node.attrib['word']+"keywordtag "	
			
							# finalresult[((nodelist.index(node))*2)] = node.attrib['word']+"keywordtag"
							# finalresult[(nodelist.index(node)*2)+1] = " "

						elif (node.attrib['id']== targetwords[0].attrib['idref']):

						#this is the target
							# finalresult[(nodelist.index(node)*2)-1] = "\t"
							# finalresult[(nodelist.index(node)*2)] = node.attrib['word']
							# finalresult[(nodelist.index(node)*2)+2] = "\t"								
							result = result + "\t"+ node.attrib['word'] 
							result = result+ " \t" 
							#print(result)

					# elif (node.attrib['id']== targetwords[0]):
					# 	#this the end of the targetwords. A two word example: White House
					# 	#NOTE: this is under the assumption that multiple words are always continuous
					# 	finalresult[(nodelist.index(node)*2)-2] = "\t"
					# 	finalresult[(nodelist.index(node)*2)-1] = node.attrib['word']				
					# 	finalresult[(nodelist.index(node)*2)] = " "	

					# elif (node.attrib['id']== targetwords[len(targetwords)-1]):
					# 	#this the end of the targetwords. A two word example: White House
					# 	finalresult[(nodelist.index(node)*2)-1] = node.attrib['word']				
					# 	finalresult[(nodelist.index(node)*2)] = "\t"	
						else:
						#this is a normal word
							# print (str((nodelist.index(node)*2)+1))
							# print (finalresult)
							# finalresult[(nodelist.index(node)*2)+1] = node.attrib['word']
							# finalresult[(nodelist.index(node)*2)+2] = " "	
							result = result + node.attrib['word']+ " "

					# item = ""
					# for item in finalresult:
					# 	result = result + item
					result = result + os.linesep
					datafile.write(result)
					used = used +1

				else:
					unused = unused + 1 			

	datafile.close()

	return [used, unused]	






#Makes or rewrites a directory to put the files in

if os.path.exists(parsed_path):
    shutil.rmtree(parsed_path)
os.makedirs(parsed_path)


corpora = [wordfile for wordfile in os.listdir(corpora_path) if wordfile != ".DS_Store"]
totalyes = 0
totalno = 0
for word in corpora:
	try:
		yesused = parse_file(word)[0]
		noused = parse_file(word)[1]	
	except Exception:
		raise Exception(word)


	print("Parsed " + word, str(noused) + " unused", str(yesused) + " used")
	totalyes = totalyes + yesused
	totalno = totalno + noused


print ("Total unused: " + str(totalno))
print ("Total used: " + str(totalyes))


# testing only
#parse_file("./rueffeln.xml")







