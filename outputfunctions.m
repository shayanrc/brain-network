1;
%Method to write category to csv file.
function initOutput(filename)
dataArray = {'image','acantharia_protist_big_center','acantharia_protist_halo','acantharia_protist','amphipods','appendicularian_fritillaridae','appendicularian_s_shape','appendicularian_slight_curve','appendicularian_straight','artifacts_edge','artifacts','chaetognath_non_sagitta','chaetognath_other','chaetognath_sagitta','chordate_type1','copepod_calanoid_eggs','copepod_calanoid_eucalanus','copepod_calanoid_flatheads','copepod_calanoid_frillyAntennae','copepod_calanoid_large_side_antennatucked','copepod_calanoid_large','copepod_calanoid_octomoms','copepod_calanoid_small_longantennae','copepod_calanoid','copepod_cyclopoid_copilia','copepod_cyclopoid_oithona_eggs','copepod_cyclopoid_oithona','copepod_other','crustacean_other','ctenophore_cestid','ctenophore_cydippid_no_tentacles','ctenophore_cydippid_tentacles','ctenophore_lobate','decapods','detritus_blob','detritus_filamentous','detritus_other','diatom_chain_string','diatom_chain_tube','echinoderm_larva_pluteus_brittlestar','echinoderm_larva_pluteus_early','echinoderm_larva_pluteus_typeC','echinoderm_larva_pluteus_urchin','echinoderm_larva_seastar_bipinnaria','echinoderm_larva_seastar_brachiolaria','echinoderm_seacucumber_auricularia_larva','echinopluteus','ephyra','euphausiids_young','euphausiids','fecal_pellet','fish_larvae_deep_body','fish_larvae_leptocephali','fish_larvae_medium_body','fish_larvae_myctophids','fish_larvae_thin_body','fish_larvae_very_thin_body','heteropod','hydromedusae_aglaura','hydromedusae_bell_and_tentacles','hydromedusae_h15','hydromedusae_haliscera_small_sideview','hydromedusae_haliscera','hydromedusae_liriope','hydromedusae_narco_dark','hydromedusae_narco_young','hydromedusae_narcomedusae','hydromedusae_other','hydromedusae_partial_dark','hydromedusae_shapeA_sideview_small','hydromedusae_shapeA','hydromedusae_shapeB','hydromedusae_sideview_big','hydromedusae_solmaris','hydromedusae_solmundella','hydromedusae_typeD_bell_and_tentacles','hydromedusae_typeD','hydromedusae_typeE','hydromedusae_typeF','invertebrate_larvae_other_A','invertebrate_larvae_other_B','jellies_tentaclesv','polychaete','protist_dark_center','protist_fuzzy_olive','protist_noctiluca','protist_other','protist_star','pteropod_butterfly','pteropod_theco_dev_seq','pteropod_triangle','radiolarian_chain','radiolarian_colony','shrimp_caridean','shrimp_sergestidae','shrimp_zoea','shrimp-like_other','siphonophore_calycophoran_abylidae','siphonophore_calycophoran_rocketship_adult','siphonophore_calycophoran_rocketship_young','siphonophore_calycophoran_sphaeronectes_stem','siphonophore_calycophoran_sphaeronectes_young','siphonophore_calycophoran_sphaeronectes','siphonophore_other_parts','siphonophore_partial','siphonophore_physonect_young','siphonophore_physonect','stomatopod','tornaria_acorn_worm_larvae','trichodesmium_bowtie','trichodesmium_multiple','trichodesmium_puff','trichodesmium_tuft','trochophore_larvae','tunicate_doliolid_nurse','tunicate_doliolid','tunicate_partial','tunicate_salp_chains','tunicate_salp','unknown_blobs_and_smudges','unknown_sticks','unknown_unclassified'};
cell2csv(filename,dataArray);
endfunction

%Method to write image name along with category array to csv file.
function writeOutput(fname,imageName,cat)

dataOtuputCellArray = {imageName};
[row column] = size(cat);
  for index = 1:column
    dataOtuputCellArray(index+1) = cat(index);
  endfor

cell2csv(fname,dataOtuputCellArray);
endfunction


% Method to write cell array content into a *.csv file.
function cell2csv(filename,cellArray)
delimiter = ',';

datei = fopen(filename,'a');
for z=1:size(cellArray,1)
    for s=1:size(cellArray,2)

        var = eval(['cellArray{z,s}']);

        if size(var,1) == 0
            var = '';
        endif

        if isnumeric(var) == 1
            var = num2str(var);
        endif

        fprintf(datei,var);

        if s ~= size(cellArray,2)
            fprintf(datei,[delimiter]);
        endif
 endfor
    fprintf(datei,'\n');
endfor
fclose(datei);
endfunction