sliderTime  = 0
for t = 0 to 198 do (
local newName = "E:\\Deployments\\bin\\obj\\Animation\\InPlace\\InPlace_" + (t as string) + ".obj"
exportFile newName #noPrompt selectedOnly:false using:ObjExp	
sliderTime += 1
)
 