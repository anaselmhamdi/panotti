#!/usr/bin/env python3
'''
SortingHatApp  - Desktop version of Sorting H.A.T. program.

Author: Scott H. Hawley @drscotthawley

TODO:
    - Everything.  Still just a facade, doesn't actually work at all.
    - Just learning Kivy as I write this. Still quite confused.
    - Should create a "ButtonBarAndStatus" class that can be reused multiple times

Requirements:
    $ pip install kivy kivy-garden scandir functools paramiko git+https://github.com/jbardin/scp.py.git
    $ garden install filebrowser
============
'''

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.filebrowser import FileBrowser
from kivy.config import Config
from settingsjson import settings_json
import time
import subprocess
import threading
from scp_upload import scp_upload
from functools import partial
import os

PANOTTI_HOME = os.path.expanduser("~")+"/panotti"
PREPROC_DIR = "Preproc"
ARCHIVE_NAME = PREPROC_DIR+".tar.gz"

def count_files(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len(files)
    return total


def folder_size(path):   # get bytes
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += folder_size(entry.path)
    return total


# check to see if a thread is still running, and if not call completion
def check_for_completion(thread, completion, t):
    if not thread.isAlive():
        completion()
        return False    # cancels any Kivy scheduler
    return True         # keep the scheduler going

# Generic utility to run things ('threads' or 'processes') in the background
#   routine can be a string (for shell command) or another python function
#   progress and oompletion are callbacks, i.e. should point to other functions
#  Note: the progress callback is actually what handles the completion
def spawn(routine, progress=None, interval=0.1, completion=None):

    def runProcInThread(cmd, completion_in_thread):  # cmd is a string for a shell command, usually completion_in_thread will be None
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()                          # make sure it's finished
        if (completion_in_thread is not None):
            completion_in_thread()
        return

    # spawn the process/thread
    if isinstance(routine, str):  # routine is a string, spawn a shell command
        thread = threading.Thread(target=runProcInThread, args=(routine,None) )
    elif callable(routine):       # routine is another Python function
        thread = threading.Thread(target=routine)
    else:
        print(" Error: routine = ",routine," is neither string nor callable")
        return           # Leave
    thread.start()

    # schedule a Kivy clock event to repeatedly call the progress-query routine (& check for completion of process)
    if (progress is not None):
        progress_clock_sched = Clock.schedule_interval(partial(progress, thread, completion), interval)
    elif (completion is not None):          # no progress per se, but still wait on completion
        completion_clock_sched = Clock.schedule_interval(partial(check_for_completion, thread, completion), interval)

    return


Builder.load_string("""
#:import expanduser os.path.expanduser

<CustomWidthTabb@TabbedPanelItem>
    width: self.texture_size[0]
    padding: 20, 0
    size_hint_x: None

<SHPanels>:
    id: SH_widget
    size_hint: 1,1
    do_default_tab: False
    tab_width: None
    CustomWidthTabb:
        id: trainPanel
        text: 'Train the Neural Net'
        BoxLayout:
            orientation: 'vertical'
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: 'Select Samples Folder'
                    on_release: SH_widget.show_load()
                Label:
                    id: samplesDir
                    text: 'No folder selected'
                    size: self.texture_size
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Start Server"
                    id: serverButton
                    on_release: root.start_prog_anim('serverProgress')
                ProgressBar:
                    id: serverProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Create Spectrograms"
                    on_release: root.preproc('preprocProgress')
                ProgressBar:
                    id: preprocProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Upload"
                    id: uploadButton
                    on_release: root.upload('uploadProgress')
                ProgressBar:
                    id: uploadProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Train"
                    id: trainButton
                    on_release: root.train('trainProgress')
                ProgressBar:
                    id: trainProgress
                    value: 0
            Label:
                id: statusMsg
                text: "Status: Initial State"
                center: self.parent.center

    CustomWidthTabb:
        text: 'Sort Your Library'
        id: sortPanel
        BoxLayout:
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: 'Select Files to Sort'
                    size_hint_y: 0.1
                    on_release: SH_widget.show_load()
                ScrollView:
                    Label:
                        id: sortFilesDisplay
                        text: 'No Files selected'
            Button:
                text: 'Go!'
                on_release: root.test_spawn()
    CustomWidthTabb:
        text: 'About'
        BoxLayout:
            RstDocument:
                text:
                    '\\n'.join(("About", "-----------",
                    "Sorting H.A.T.* - Organize your audio library with the help of neural nets.\\n",
                    "Built on `Panotti <http://github.com/drscotthawley/panotti>`_ by @drscotthawley"))

            Image:
                source: 'static/sorting-hat-logo.png'
                canvas.before:
                    Color:
                        rgba: .9, .9, .9, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

    CustomWidthTabb:
        id: settingsPanel
        text: 'Settings'
        on_press: SH_widget.my_handle_settings()
        Button:
            text: 'Press to go to Train'
            on_release: root.switch_to(trainPanel)



<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileBrowser:
            id: filechooser
            multiselect: True
            dirselect: True
            path: expanduser("~")
            on_canceled: root.cancel()
            on_success: root.load(filechooser.path, filechooser.selection)
""")

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


#============== Main Widget =================

class SHPanels(TabbedPanel):
    def __init__(self, **kwargs):
        super(SHPanels, self).__init__(**kwargs)
        Clock.schedule_once(self.on_tab_width, 0.1)
        Window.bind(on_dropfile=self._on_file_drop)
        self.last_drop_time = time.time()-10000
        self.progress_events = {}
        self.samplesDir = ''
        self.parentDir = ''
        self.ready_to_preproc = False
        self.ready_to_upload = False
        self.ready_to_train = False

    #----------- Testing stuffffff-------
        self.count = 0
        self.maxval = 0

    def finished(self):
        print("\n\n*** Finished.")



    def count_to(self, maxval):   # this will be our test process
        print("Begin the counting, maxval = ",maxval)
        self.maxval = maxval
        for self.count in range(maxval):
            pass

    def test_spawn(self):
        spawn(partial(self.count_to,50000000), progress=self.progress_display, completion=self.finished)
        #time.sleep(10)  # just keep the program from terminating so we can see what happens!

    #-------------- Generic Utilities for testing / mock-up  ------------
    def done_fake(self):
        self.ids['statusMsg'].text = "Server is up and running."

    def progress_display(self, barname, thread, completion, t):
        percent = int((self.count+1) / self.maxval * 100)
        self.ids[barname].value = percent
        # Test for completion:
        if not thread.isAlive():         # if the thread has completed
            if (percent >=100):          # Yay
                completion()    # go on to the final state
            else:
                print("\nError: Process died but progress < 100% ")
            return False                 # Either way, cancel the Kivy clock schedule
        return True                      # keep the progress-checker rolling

    def count_up(self,maxval=100):     # some 'fake' task for testing purposes
        self.maxval = maxval
        for self.count in range(maxval):
            time.sleep(0.02)


    def start_prog_anim(self,barname):
        self.ready_to_preproc = ('No folder selected' != self.ids['samplesDir'].text)
        self.ready_to_upload = (self.ids['preprocProgress'].value >= 100) and (self.ids['serverProgress'].value >= 100)
        self.ready_to_train = (self.ids['uploadProgress'].value >= 100) and (self.ready_to_upload)
        if (('serverProgress' == barname) or
            (('preprocProgress' == barname) and self.ready_to_preproc) or
            (('uploadProgress' == barname) and self.ready_to_upload) or
            (('trainProgress' == barname) and self.ready_to_train) ):
            #self.ids[barname].value = 0
            spawn(self.count_up, progress=partial(self.progress_display,barname),interval=0.1, completion=self.done_fake)


    #-------------- File/Folder Selection --------------

    # TODO: does this get used?  copied code from elsewhere
    def open(self, path, filename):
        with open(os.path.join(path, filename[0])) as f:
            print(f.read())

    # TODO: does this get used?  copied code from elsewhere
    def selected(self, filename):
        print("selected: %s" % filename[0])

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    # either by drag-dropping or by using popup dialog, we now have one or more filenames/directory names
    def got_filenames(self, filenames):
        if (self.current_tab == self.ids['trainPanel']):
            self.samplesDir = filenames[0]
            self.parentDir = self.samplesDir+'/../'
            self.ids['samplesDir'].text = self.samplesDir
            self.totalClips = count_files(self.samplesDir)
            text = "Contains "+str(self.totalClips)+" files"
            self.ids['statusMsg'].text = text
        elif (self.current_tab == self.ids['sortPanel']):
            self.sortFileList = filenames
            self.ids['sortFilesDisplay'].text  = ''
            for name in filenames:
                self.ids['sortFilesDisplay'].text += str(name) + '\n'
        return

    # this doesn't actually load file, it's just the result of the file selector gui
    def load(self, path, filenames):
        self.dismiss_popup()
        if (filenames):
            self.got_filenames(filenames )

    # sometimes you just want a widget id (name)
    def get_id(self,  instance):
        for id, widget in instance.parent.ids.items():
            if widget.__self__ == instance:
                return id


    # if you drag & drop multiple files, it treats them as separate events; but we want one list o files
    def consolidate_drops(self, file_path):
        now = time.time()
        tolerance = 1
        if (now - self.last_drop_time > tolerance):
            self.sortFileList=[file_path]
            self.ids['sortFilesDisplay'].text = file_path
        else:
            self.sortFileList.append(file_path)
            self.ids['sortFilesDisplay'].text += '\n'+file_path
        self.last_drop_time = now
        print("Sort!  self.sortFileList = ",self.sortFileList)

    # this fires multiple times if multiple files are dropped
    def _on_file_drop(self, window, file_path):
        if (self.current_tab == self.ids['trainPanel']):
            self.got_filenames( [file_path.decode('UTF-8')] )
        elif (self.current_tab == self.ids['sortPanel']):
            self.consolidate_drops(file_path.decode('UTF-8'))


    #-------------- Preprocessing --------------

    def monitor_preproc(self, folder, thread, completion, dt):
        files_processed = count_files(folder)      # Folder should be Preproc
        self.ids['preprocProgress'].value = max(3, int(files_processed / self.totalClips * 100))
        self.ids['statusMsg'].text = str(files_processed)+"/"+str(self.totalClips)+" files processed"
        if (self.ids['preprocProgress'].value >= 99.4):
            return False              # this just cancels the clock schedule
        return

    def preproc(self,barname):
        if ('' != self.samplesDir) and ('' != self.parentDir):
            self.ready_to_preproc = True
        if self.ready_to_preproc:
            cmd = 'cd '+self.parentDir+'; rm -rf '+PREPROC_DIR
            p = subprocess.call(cmd, shell=True)                       # blocking
            cmd = 'cd '+self.parentDir+'; '+PANOTTI_HOME+'/preprocess_data.py '
            if App.get_running_app().config.get('example', 'sequential'):
                cmd += '-s '
            if App.get_running_app().config.get('example', 'mono'):
                cmd += '-m '
            cmd += '--dur='+App.get_running_app().config.get('example','duration')+' '
            cmd += '-r='+App.get_running_app().config.get('example','sampleRate')+' '
            cmd += '--format='+App.get_running_app().config.get('example','specFileFormat')+' '
            cmd += ' | tee log.txt '
            print('Executing command: ',cmd)
            spawn(cmd, progress=partial(self.monitor_preproc,self.parentDir+PREPROC_DIR), interval=0.2, completion=None )
        return


    #-------------- Uploading --------------
    # for progress purposes, we'll split the percentage 40/60 between archive & upload

    # status messages , progress and such
    def my_upload_callback(self, filename, size, sent):
        percent = 60 + int( float(sent)/float(size)*60)
        prog_str = 'Uploading progress: '+str(percent)+' %'
        self.ids['statusMsg'].text = prog_str
        barname = 'uploadProgress'
        self.ids[barname].value = percent

    # TODO: decide on API for file transfer. for now, we use scp
    def actual_upload(self, archive_path):
        self.server = App.get_running_app().config.get('example', 'server')
        self.username = App.get_running_app().config.get('example', 'username')
        scp_upload( src_blob=archive_path, options={'hostname': self.server, 'username': self.username}, progress=self.my_upload_callback )

    # Watches progress of packaging the Preproc/ directory.
    def monitor_archive(self, archive_file, orig_size, thread, completion, dt):
        #TODO: ok this is sloppy but estimating compression is 'hard'; we generally get around a factor of 10 in compression
        if (os.path.isfile(archive_file) ):  # problem with this is, zip uses an alternate name until it's finished
            archive_size = os.path.getsize(archive_file)
            est_comp_ratio = 8
            est_size = orig_size/est_comp_ratio
            percent = int(archive_size / est_size * 50)
            self.ids['statusMsg'].text = "Archiving... "+str(percent)+" %"

        if not thread.isAlive():           # archive process completed
            if (percent < 99):
                print("  Warning: archive finished with less than 100% complete")
            self.ids['statusMsg'].text = "Now Uploading..."
            completion()
            return False            # cancels scheduler
        return

    # this actually initiates "archiving" (zip/tar) first, and THEN uploads
    def upload(self,barname):
        archive_path =  self.parentDir+ARCHIVE_NAME
        self.ready_to_upload = os.path.exists(archive_path) and (self.ids['preprocProgress'].value >= 100) and (self.ids['serverProgress'].value >= 100)
        if (self.ready_to_upload):
            self.ids['statusMsg'].text = "Archiving Preproc...)"
            cmd = 'cd '+self.parentDir+'; rm -f '+archive_path+';  tar cfz '+archive_path+' Preproc/'
            orig_size = folder_size(self.parentDir+'Preproc')
            spawn(cmd, progress=partial(self.monitor_archive,archive_path,orig_size), interval=0.2, completion=partial(self.actual_upload,archive_path) )
        return


    #-------------- Training --------------
    def train_is_complete(self):
        self.ids['statusMsg'].text = 'Training is complete!'
        return False    # cancel clock schedule

    def download_weights(self):
        print("\n\nDownloading weights...")
        if ('' == self.parentDir):
            dst = "~/Downloads/"
        else:
            dst = self.parentDir
        cmd = "scp "+self.username+'@'+self.server+':weights.hdf5 '+dst
        print("Executing command cmd = [",cmd,"]")
        spawn(cmd, progress=None, completion=self.train_is_complete)
        return

    def train_progress(self,thread, completion, t):
        if not thread.isAlive():
            print("Train thread finished, calling completion (download weights?)...")
            completion()
            return False    # cancel schedule

    def train(self, barname, method='ssh'):
        self.ready_to_train = True# (self.ids['uploadProgress'].value >= 100) and (self.ready_to_upload)
        if self.ready_to_train:
            self.server = App.get_running_app().config.get('example', 'server')
            self.username = App.get_running_app().config.get('example', 'username')

            if ('ssh' == method):
            # remote code execution via SSH server. could use sorting-hat HTTP server instead
                cmd = 'ssh -t '+self.username+'@'+self.server+' "tar xvfz Preproc.tar.gz;'
                cmd += ' ~/panotti/train_network.py'
                cmd += ' --epochs='+App.get_running_app().config.get('example','epochs')
                cmd += ' --val='+App.get_running_app().config.get('example','val_split')
                cmd += ' --format='+App.get_running_app().config.get('example','specFileFormat')
                cmd += ' | tee log.txt"'
                print("Executing command cmd = [",cmd,"]")

                spawn(cmd, progress=self.train_progress, interval=1, completion=self.download_weights)
                #p = subprocess.call(cmd, shell=True)   # blocking  TODO: make non-blocking
            elif ('http' == method):
                print("Yea, haven't done that yet")
            else:
                print("Error: Unrecognized API method '",method,"''")


    #-------------- Settings --------------

    # programaticaly change to tab_state (i.e. tab instance)
    def change_to_tab(self, tab_state, t):
        self.switch_to(tab_state)

    # opens settings view, returns to tab you were on before
    def my_handle_settings(self):
        tab_state = self.current_tab            # remember what tab we're on
        App.get_running_app().open_settings()
        Clock.schedule_once(partial(self.change_to_tab, tab_state), 0.1)  # switch back to orig tab after a slight delay

#============== End of main widget ==============

class SortingHatApp(App):
    def build(self):
        self.icon = 'static/sorting-hat-logo.png'
        self.use_kivy_settings = False
        return SHPanels()

    def build_config(self, config):
        config.setdefaults('example', {
            'server': 'lecun',
            'username': os.getlogin(),      # default is that they have the same username on both local & server
            'sshKeyPath': '~/.ssh/id_rsa.pub',
            'mono': True,
            'sequential': True,
            'duration': 3,
            'sampleRate': 44100,
            'specFileFormat': 'png',   # note, color png supports only up to 4 channels of audio, npy is arbitrarily many, jpeg is lossy
            'weightsOption': 'Default',
            'server': 'lecun.belmont.edu',
            'sshKeyPath': '~/.ssh/id_rsa.pub',
            'epochs': 20,
            'val_split': 0.1,
            })

    def build_settings(self, settings):
        settings.add_json_panel('Settings',
                                self.config,
                                data=settings_json)

    def on_config_change(self, config, section,
                         key, value):
        print(config, section, key, value)


if __name__ == '__main__':
    SortingHatApp().run()
