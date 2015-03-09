1;

%==============Functions for geting inputs==============%

function ImgInputMat=makeImgInputs(a,limit)
  startRow=1;
  startCol=1;
  [maxX maxY]=size(a); 
  ImgInputMat=zeros(1,limit^2);
  endRow=startRow+limit-1;
  endCol=startCol+limit-1;
  while endRow<=maxX
    while endCol<=maxY
      %disp([startRow,endRow,startCol,endCol])
      submatrix=a(startRow:endRow,startCol:endCol);
      matArray=[submatrix'(:)']; %flatten the submatrix
      ImgInputMat=[ImgInputMat;matArray]; % and make it a row
      startCol+=(limit/2);
      endCol=startCol+limit-1;
    endwhile
    startCol=1;
    endCol=startCol+limit-1;
    startRow+=(limit/2);    
    endRow=startRow+limit-1;
  endwhile
  ImgInputMat=ImgInputMat(2:end,:); %remove the 1st row with zeros
endfunction

%==============Functions for Feature Scaling==============%
%function to normalize input values to between -1 & 1
function normalizedInputs = normalize(inputs)
  flatInputs=inputs(:);
  inputsRange=range(flatInputs);
  midRange=inputsRange/2;
  normalizedInputs=(inputs-midRange)/inputsRange;
endfunction  



function finalPosArr=getImageCategory(imagePath)
  index = 0;
  %dataArray = textread('Data.txt','%s');
  dataArray = {'acantharia_protist_big_center','Photos','acantharia_protist_halo','acantharia_protist','amphipods','appendicularian_fritillaridae','appendicularian_s_shape','appendicularian_slight_curve','appendicularian_straight','artifacts_edge','artifacts','chaetognath_non_sagitta','chaetognath_other','chaetognath_sagitta','chordate_type1','copepod_calanoid_eggs','copepod_calanoid_eucalanus','copepod_calanoid_flatheads','copepod_calanoid_frillyAntennae','copepod_calanoid_large_side_antennatucked','copepod_calanoid_large','copepod_calanoid_octomoms','copepod_calanoid_small_longantennae','copepod_calanoid','copepod_cyclopoid_copilia','copepod_cyclopoid_oithona_eggs','copepod_cyclopoid_oithona','copepod_other','crustacean_other','ctenophore_cestid','ctenophore_cydippid_no_tentacles','ctenophore_cydippid_tentacles','ctenophore_lobate','decapods','detritus_blob','detritus_filamentous','detritus_other','diatom_chain_string','diatom_chain_tube','echinoderm_larva_pluteus_brittlestar','echinoderm_larva_pluteus_early','echinoderm_larva_pluteus_typeC','echinoderm_larva_pluteus_urchin','echinoderm_larva_seastar_bipinnaria','echinoderm_larva_seastar_brachiolaria','echinoderm_seacucumber_auricularia_larva','echinopluteus','ephyra','euphausiids_young','euphausiids','fecal_pellet','fish_larvae_deep_body','fish_larvae_leptocephali','fish_larvae_medium_body','fish_larvae_myctophids','fish_larvae_thin_body','fish_larvae_very_thin_body','heteropod','hydromedusae_aglaura','hydromedusae_bell_and_tentacles','hydromedusae_h15','hydromedusae_haliscera_small_sideview','hydromedusae_haliscera','hydromedusae_liriope','hydromedusae_narco_dark','hydromedusae_narco_young','hydromedusae_narcomedusae','hydromedusae_other','hydromedusae_partial_dark','hydromedusae_shapeA_sideview_small','hydromedusae_shapeA','hydromedusae_shapeB','hydromedusae_sideview_big','hydromedusae_solmaris','hydromedusae_solmundella','hydromedusae_typeD_bell_and_tentacles','hydromedusae_typeD','hydromedusae_typeE','hydromedusae_typeF','invertebrate_larvae_other_A','invertebrate_larvae_other_B','jellies_tentaclesv','polychaete','protist_dark_center','protist_fuzzy_olive','protist_noctiluca','protist_other','protist_star','pteropod_butterfly','pteropod_theco_dev_seq','pteropod_triangle','radiolarian_chain','radiolarian_colony','shrimp_caridean','shrimp_sergestidae','shrimp_zoea','shrimp-like_other','siphonophore_calycophoran_abylidae','siphonophore_calycophoran_rocketship_adult','siphonophore_calycophoran_rocketship_young','siphonophore_calycophoran_sphaeronectes_stem','siphonophore_calycophoran_sphaeronectes_young','siphonophore_calycophoran_sphaeronectes','siphonophore_other_parts','siphonophore_partial','siphonophore_physonect_young','siphonophore_physonect','stomatopod','tornaria_acorn_worm_larvae','trichodesmium_bowtie','trichodesmium_multiple','trichodesmium_puff','trichodesmium_tuft','trochophore_larvae','tunicate_doliolid_nurse','tunicate_doliolid','tunicate_partial','tunicate_salp_chains','tunicate_salp','unknown_blobs_and_smudges','unknown_sticks','unknown_unclassified'};
  finalPosArr= zeros(size(dataArray));
  dirPath = fileparts(imagePath);
  parentFolder = strsplit(dirPath,"/");
  parentFolder = parentFolder(end);
  str = parentFolder{1,end};
  index = find(cellfun(@(x) strcmp(x,str), dataArray));
  %finalPosArr = initArr(:,indices)'
  finalPosArr(index)=1;
endfunction


function dataElementStruct = getTrainingSet(dirName)
  dirData = dir(dirName)  ;    %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  %fileList = {dirData(~dirIndex).name}'  ;%'# Get a list of the files
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'}); %# Find index of subdirectories
                                            %#   that are not '.' or '..'
  dataStructArr = zeros(1);
  dataElementStruct = struct();
  counter = 1;
    for iDir = find(validIndex)                  %# Loop over valid subdirectories
      nextDir = fullfile(dirName,subDirs{iDir}) ;  %# Get the subdirectory path
      dirData = dir(nextDir);
      fileIndex = [dirData.isdir];
      fileList = {dirData(~fileIndex).name};
      
      %fileList = [getDirData(nextDir)];
      for index = 1:length(fileList)
        
        fileName = fileList{index};
        disp(sprintf("generating trainingset for file %s \n",fileName));
        dirPath = file_in_loadpath(nextDir);
        filePath = strcat(dirPath,filesep(),fileName);
        finalPosArr = getImageCategory(filePath); 
        %imageArr = getImageArr(filePath);
        dataElementStruct(counter).categoryArr = finalPosArr;
        dataElementStruct(counter).fileDir = filePath;
        dataElementStruct(counter).imageName = fileName;
        %dataElement = getDataStructEle(dataStructTemp,finalPosArr,imageArr,nextDir,fileName);
        %dataStructArr = [dataStructArr;{dataElement}];
        counter ++;
      endfor
      
     endfor
     
endfunction

function dataStructTemp = getDataStructEle(dataStructTemp,finalPosArr,imageArr,nextDir,fileName)
      
      
      dataStructTemp.posArr = finalPosArr;     % dataStructTemp.imageArr = imageArr;
      dataStructTemp.fileDir = nextDir;
      dataStructTemp.imageName = fileName;
endfunction

function imageArr = getImageArr(filePath)
imageArr = imread(filePath);
endfunction