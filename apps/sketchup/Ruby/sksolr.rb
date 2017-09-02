require 'sketchup.rb'

module SkSolR
# Constants
# --------------------------------------------------------------------------------
COMPANY_NAME="Fast And Serious"
PRODUCT_NANE="SoL-R"
OBJ_FOLDER="obj"
OBJ_FILENAME="SU.obj"
SOLR_CUDA_EXE="SoL_R_Editor_Cuda.exe"
SOLR_OPENCL_EXE="SoL_R_Editor_OpenCL.exe"
SOLR_RESOLUTION="-width=512 -height=388"
VERSION="1.0.00.0"

path = File.dirname( __FILE__ )
APPLICATION_FOLDER=ENV['ProgramW6432']+'\\'+COMPANY_NAME+'\\'+PRODUCT_NANE
USER_FOLDER=ENV['USERPROFILE']+'\Fast And Serious'

# Generate user folders
# --------------------------------------------------------------------------------
Dir.mkdir( USER_FOLDER ) unless File.exists?(USER_FOLDER)

USER_FOLDER = USER_FOLDER + '\\' + PRODUCT_NANE
Dir.mkdir( USER_FOLDER ) unless File.exists?(USER_FOLDER)

USER_FOLDER = USER_FOLDER+'\\'+OBJ_FOLDER
Dir.mkdir( USER_FOLDER ) unless File.exists?(USER_FOLDER)

# Add a menu item to launch our plugin.
# --------------------------------------------------------------------------------
toolbar = UI::Toolbar.new PRODUCT_NANE      

# CUDA      
# --------------------------------------------------------------------------------
cmd_cuda = UI::Command.new("Start "+PRODUCT_NANE) {
	Dir.chdir( APPLICATION_FOLDER )
	Thread.new {system(APPLICATION_FOLDER+'\\'+SOLR_CUDA_EXE+' -objFile="' + USER_FOLDER+'\\'+OBJ_FILENAME+'" '+SOLR_RESOLUTION)}
}
cmd_cuda.small_icon = "su_solr/Images/SoL_R.jpg"
cmd_cuda.large_icon = "su_solr/Images/SoL_R.jpg"
cmd_cuda.tooltip = "Run CUDA visualizer"
cmd_cuda.status_bar_text = "Run CUDA visualizer"
cmd_cuda.menu_text = "Run CUDA visualizer"

# OpenCL      
# --------------------------------------------------------------------------------
cmd_opencl = UI::Command.new("Start "+PRODUCT_NANE) {         
	Dir.chdir( APPLICATION_FOLDER )
	Thread.new {system(APPLICATION_FOLDER+'\\'+SOLR_OPENCL_EXE+' -objFile="' + USER_FOLDER+'\\'+OBJ_FILENAME+'" '+SOLR_RESOLUTION)}
}
cmd_opencl.small_icon = "su_solr/Images/SoL_R_OpenCL.jpg"
cmd_opencl.large_icon = "su_solr/Images/SoL_R_OpenCL.jpg"
cmd_opencl.tooltip = "Run OpenCL visualizer"
cmd_opencl.status_bar_text = "Run OpenCL visualizer"
cmd_opencl.menu_text = "Run OpenCL visualizer"

# Export to OBJ
# --------------------------------------------------------------------------------
cmd_export = UI::Command.new("Export model to "+PRODUCT_NANE) {         
	model = Sketchup.active_model
	options_hash = { :triangulated_faces   => true,
					  :doublesided_faces    => true,
					  :edges                => false,
					  :materials_by_layer   => false,
					  :author_attribution   => false,
					  :texture_maps         => true,
					  :selectionset_only    => false,
					  :preserve_instancing  => true }
	model.export USER_FOLDER+'\\'+OBJ_FILENAME,false
}
cmd_export.small_icon = "su_solr/Images/SoL_R_Export.jpg"
cmd_export.large_icon = "su_solr/Images/SoL_R_Export.jpg"
cmd_export.tooltip = "Export model to OBJ"
cmd_export.status_bar_text = "Export model to OBJ"
cmd_export.menu_text = "Export model to OBJ"

# Toolbar
# --------------------------------------------------------------------------------
toolbar = toolbar.add_item cmd_cuda
toolbar = toolbar.add_item cmd_opencl
toolbar = toolbar.add_item cmd_export
toolbar.show

end
