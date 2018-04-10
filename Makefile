
inx = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita/platealign.inx
py  = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita/platealign.py 
files = $(inx) $(py)

targetdir = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita

share: $(files)


$(inx): $(notdir $(inx))
	cp $^ $@

$(py): $(notdir $(py))
	cp $^ $@
