
inx = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita/platealign.inx
py  = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita/platealign.py 

inx3 = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita/platealign3.inx
py3  = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita/platealign3.py 

files = $(inx) $(py)

targetdir = /home/yfujita/share/lab/Meeting/weekly_meeting2016.2-/yfujita

share: $(files)


$(inx): $(notdir $(inx))
	cp $^ $@

$(py): $(notdir $(py))
	cp $^ $@
