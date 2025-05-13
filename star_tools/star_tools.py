import pandas as pd
import numpy as np


# Get header of star file. mode='old' is for old star file format.
def _get_header(star_file, mode='old'):
    if mode == 'new':
        return _get_header_new(star_file, 'new')
    nHeaderLines = 0  # Number of header lines
    with open(star_file, 'r') as f:
        star = f.readlines()
    labels = []
    for line in star:  # Go through each line of star file untill you meet the first non column line
        nHeaderLines += 1
        if line.startswith('_rln') or line.startswith('_ps'):  # Column labels
            # add star column name to labels list
            labels.append(line.split()[0])
        elif labels:  # First non column line breaks the loop.
            break
    return ([], [], labels, star[nHeaderLines-1:])


# Get header of star file. mode='new' is for
def _get_header_new(star_file, mode='new'):
    if mode == 'old':
        return _get_header(star_file, 'old')
    nMetaLines = 0
    with open(star_file, 'r') as f:
        star = f.readlines()
    metaLabels = []
    metaInformation = []
    labels = []
    mode = ''
    for line in star:  # Go through each line in star file untill the first non-meta column name line is found
        nMetaLines += 1
        if line.startswith('_rln') or line.startswith('_ps'):
            mode = 'metaLabels'
            params = line.split()
            metaLabels.append(params[0])
        elif mode == 'metaLabels':  # Breaks after first non column name line, after starting going through metalabels
            break
    star = star[nMetaLines-1:]
    for line in star:
        if line.strip():
            metaInformation.append([x for x in line.split(' ') if x != ""])
        else:
            break
    nHeaderLines = 0  # Number of header lines
    for line in star:
        nHeaderLines += 1
        if line.startswith('_rln') or line.startswith('_ps'):  # Column labels
            mode = 'labels'
            params = line.split()
            labels.append(params[0])  # add star column name to labels list
        elif mode == 'labels':  # start of data break loop
            break
    data = star[nHeaderLines-1:]
    return (metaLabels, metaInformation, labels, data)


# To extract a column and convert it a certain type (float, string, int, etc.)
def _get_columns(starData, columnName='_rlnUnknownLabel', convertType=False):
    if not (columnName.startswith('_rln', 0, 4) or columnName.startswith('_ps')):
        columnName = f'_rln{columnName}'
        print(f'Column name renamed to {columnName}')
    if convertType != False:
        starData[columnName] = starData[columnName].astype(convertType)
    return starData[columnName]

# uses get_columns to clean the most common columns - addional custom columns can be added though this is not recommended
def _cleanColumns(starData, addLabel=False, labelType=False):
    labelList = {
        "_rlnAccumMotionEarly": float,
        "_rlnAccumMotionLate": float,
        "_rlnAccumMotionTotal": float,
        "_rlnAccuracyRotations": float,
        "_rlnAccuracyTranslations": float,
        "_rlnAccuracyTranslationsAngst": float,
        "_rlnAdaptiveOversampleFraction": float,
        "_rlnAdaptiveOversampleOrder": int,
        "_rlnAmplitudeContrast": float,
        "_rlnAmplitudeCorrelationMaskedMaps": float,
        "_rlnAmplitudeCorrelationUnmaskedMaps": float,
        "_rlnAnglePsi": float,
        "_rlnAnglePsiFlip": bool,
        "_rlnAnglePsiFlipRatio": float,
        "_rlnAnglePsiPrior": float,
        "_rlnAngleRot": float,
        "_rlnAngleRotFlipRatio": float,
        "_rlnAngleRotPrior": float,
        "_rlnAngleTilt": float,
        "_rlnAngleTiltPrior": float,
        "_rlnAngstromResolution": float,
        "_rlnAreaId": int,
        "_rlnAreaName": str,
        "_rlnAutoLocalSearchesHealpixOrder": int,
        "_rlnAutopickFigureOfMerit": float,
        "_rlnAvailableMemory": float,
        "_rlnAverageNrOfFrames": int,
        "_rlnAveragePmax": float,
        "_rlnAverageValue": float,
        "_rlnBeamTiltClass": int,
        "_rlnBeamTiltX": float,
        "_rlnBeamTiltY": float,
        "_rlnBestResolutionThusFar": float,
        "_rlnBfactorUsedForSharpening": float,
        "_rlnBodyKeepFixed": int,
        "_rlnBodyMaskName": str,
        "_rlnBodyReferenceName": str,
        "_rlnBodyRotateDirectionX": float,
        "_rlnBodyRotateDirectionY": float,
        "_rlnBodyRotateDirectionZ": float,
        "_rlnBodyRotateRelativeTo": int,
        "_rlnBodySigmaAngles": float,
        "_rlnBodySigmaOffset": float,
        "_rlnBodySigmaOffsetAngst": float,
        "_rlnBodySigmaPsi": float,
        "_rlnBodySigmaRot": float,
        "_rlnBodySigmaTilt": float,
        "_rlnBodyStarFile": str,
        "_rlnChangesOptimalClasses": float,
        "_rlnChangesOptimalOffsets": float,
        "_rlnChangesOptimalOrientations": float,
        "_rlnChromaticAberration": float,
        "_rlnClassDistribution": float,
        "_rlnClassNumber": int,
        "_rlnClassPriorOffsetX": float,
        "_rlnClassPriorOffsetY": float,
        "_rlnCoarseImageSize": int,
        "_rlnComment": str,
        "_rlnConvergenceCone": float,
        "_rlnCoordinateX": float,
        "_rlnCoordinateY": float,
        "_rlnCoordinateZ": float,
        "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps": float,
        "_rlnCorrelationFitGuinierPlot": float,
        "_rlnCtfAstigmatism": float,
        "_rlnCtfBfactor": float,
        "_rlnCtfDataAreCtfPremultiplied": bool,
        "_rlnCtfDataArePhaseFlipped": bool,
        "_rlnCtfFigureOfMerit": float,
        "_rlnCtfImage": str,
        "_rlnCtfMaxResolution": float,
        "_rlnCtfPowerSpectrum": str,
        "_rlnCtfScalefactor": float,
        "_rlnCtfValidationScore": float,
        "_rlnCtfValue": float,
        "_rlnCurrentImageSize": int,
        "_rlnCurrentIteration": int,
        "_rlnCurrentResolution": float,
        "_rlnDataDimensionality": int,
        "_rlnDataType": int,
        "_rlnDefocusAngle": float,
        "_rlnDefocusU": float,
        "_rlnDefocusV": float,
        "_rlnDetectorPixelSize": float,
        "_rlnDiff2RandomHalves": float,
        "_rlnDifferentialPhaseResidualMaskedMaps": float,
        "_rlnDifferentialPhaseResidualUnmaskedMaps": float,
        "_rlnDoAutoRefine": bool,
        "_rlnDoCorrectCtf": bool,
        "_rlnDoCorrectMagnification": bool,
        "_rlnDoCorrectNorm": bool,
        "_rlnDoCorrectScale": bool,
        "_rlnDoExternalReconstruct": bool,
        "_rlnDoFastSubsetOptimisation": bool,
        "_rlnDoHelicalRefine": bool,
        "_rlnDoIgnoreCtfUntilFirstPeak": bool,
        "_rlnDoMapEstimation": bool,
        "_rlnDoOnlyFlipCtfPhases": bool,
        "_rlnDoRealignMovies": bool,
        "_rlnDoSkipAlign": bool,
        "_rlnDoSkipRotate": bool,
        "_rlnDoSolventFlattening": bool,
        "_rlnDoSolventFscCorrection": bool,
        "_rlnDoSplitRandomHalves": bool,
        "_rlnDoStochasticEM": bool,
        "_rlnDoStochasticGradientDescent": bool,
        "_rlnDoZeroMask": bool,
        "_rlnEERGrouping": int,
        "_rlnEERUpsampling": int,
        "_rlnEnabled": bool,
        "_rlnEnergyLoss": float,
        "_rlnEstimatedResolution": float,
        "_rlnExperimentalDataStarFile": str,
        "_rlnExtReconsDataImag": str,
        "_rlnExtReconsDataReal": str,
        "_rlnExtReconsResult": str,
        "_rlnExtReconsResultStarfile": str,
        "_rlnExtReconsWeight": str,
        "_rlnFinalResolution": float,
        "_rlnFittedInterceptGuinierPlot": float,
        "_rlnFittedSlopeGuinierPlot": float,
        "_rlnFixSigmaNoiseEstimates": bool,
        "_rlnFixSigmaOffsetEstimates": bool,
        "_rlnFixTauEstimates": bool,
        "_rlnFourierCompleteness": float,
        "_rlnFourierMask": str,
        "_rlnFourierShellCorrelation": float,
        "_rlnFourierShellCorrelationCorrected": float,
        "_rlnFourierShellCorrelationMaskedMaps": float,
        "_rlnFourierShellCorrelationParticleMaskFraction": float,
        "_rlnFourierShellCorrelationParticleMolWeight": float,
        "_rlnFourierShellCorrelationUnmaskedMaps": float,
        "_rlnFourierSpaceInterpolator": int,
        "_rlnGoldStandardFsc": float,
        "_rlnGroupName": str,
        "_rlnGroupNrParticles": int,
        "_rlnGroupNumber": int,
        "_rlnGroupScaleCorrection": float,
        "_rlnHasConverged": bool,
        "_rlnHasHighFscAtResolLimit": bool,
        "_rlnHasLargeSizeIncreaseIterationsAgo": int,
        "_rlnHealpixOrder": int,
        "_rlnHealpixOrderOriginal": int,
        "_rlnHelicalCentralProportion": float,
        "_rlnHelicalKeepTiltPriorFixed": bool,
        "_rlnHelicalMaskTubeInnerDiameter": float,
        "_rlnHelicalMaskTubeOuterDiameter": float,
        "_rlnHelicalOffsetStep": float,
        "_rlnHelicalRise": float,
        "_rlnHelicalRiseInitial": float,
        "_rlnHelicalRiseInitialStep": float,
        "_rlnHelicalRiseMax": float,
        "_rlnHelicalRiseMin": float,
        "_rlnHelicalSigmaDistance": float,
        "_rlnHelicalSymmetryLocalRefinement": bool,
        "_rlnHelicalTrackLength": float,
        "_rlnHelicalTrackLengthAngst": float,
        "_rlnHelicalTubeID": int,
        "_rlnHelicalTubePitch": float,
        "_rlnHelicalTwist": float,
        "_rlnHelicalTwistInitial": float,
        "_rlnHelicalTwistInitialStep": float,
        "_rlnHelicalTwistMax": float,
        "_rlnHelicalTwistMin": float,
        "_rlnHighresLimitExpectation": float,
        "_rlnHighresLimitSGD": float,
        "_rlnIgnoreHelicalSymmetry": bool,
        "_rlnImageDimensionality": int,
        "_rlnImageId": int,
        "_rlnImageName": str,
        "_rlnImageOriginalName": str,
        "_rlnImagePixelSize": float,
        "_rlnImageSize": int,
        "_rlnImageSizeX": int,
        "_rlnImageSizeY": int,
        "_rlnImageSizeZ": int,
        "_rlnImageWeight": float,
        "_rlnIncrementImageSize": int,
        "_rlnIs3DSampling": bool,
        "_rlnIs3DTranslationalSampling": bool,
        "_rlnIsFlip": bool,
        "_rlnIsHelix": bool,
        "_rlnJobIsContinue": bool,
        "_rlnJobOptionDefaultValue": str,
        "_rlnJobOptionDirectoryDefault": str,
        "_rlnJobOptionFilePattern": str,
        "_rlnJobOptionGUILabel": str,
        "_rlnJobOptionHelpText": str,
        "_rlnJobOptionMenuOptions": str,
        "_rlnJobOptionSliderMax": float,
        "_rlnJobOptionSliderMin": float,
        "_rlnJobOptionSliderStep": float,
        "_rlnJoboptionType": int,
        "_rlnJobOptionValue": str,
        "_rlnJobOptionVariable": str,
        "_rlnJobType": int,
        "_rlnJobTypeName": str,
        "_rlnJoinHalvesUntilThisResolution": float,
        "_rlnKullbackLeiblerDivergence": float,
        "_rlnKurtosisExcessValue": float,
        "_rlnLensStability": float,
        "_rlnLocalSymmetryFile": str,
        "_rlnLogAmplitudesIntercept": float,
        "_rlnLogAmplitudesMTFCorrected": float,
        "_rlnLogAmplitudesOriginal": float,
        "_rlnLogAmplitudesSharpened": float,
        "_rlnLogAmplitudesWeighted": float,
        "_rlnLogLikeliContribution": float,
        "_rlnLogLikelihood": float,
        "_rlnLongitudinalDisplacement": float,
        "_rlnLowresLimitExpectation": float,
        "_rlnMagMat00": float,
        "_rlnMagMat01": float,
        "_rlnMagMat10": float,
        "_rlnMagMat11": float,
        "_rlnMagnification": float,
        "_rlnMagnificationCorrection": float,
        "_rlnMagnificationSearchRange": float,
        "_rlnMagnificationSearchStep": float,
        "_rlnMaskName": str,
        "_rlnMatrix_1_1": float,
        "_rlnMatrix_1_2": float,
        "_rlnMatrix_1_3": float,
        "_rlnMatrix_2_1": float,
        "_rlnMatrix_2_2": float,
        "_rlnMatrix_2_3": float,
        "_rlnMatrix_3_1": float,
        "_rlnMatrix_3_2": float,
        "_rlnMatrix_3_3": float,
        "_rlnMaximumCoarseImageSize": int,
        "_rlnMaximumValue": float,
        "_rlnMaxNumberOfPooledParticles": int,
        "_rlnMaxValueProbDistribution": float,
        "_rlnMicrographBinning": float,
        "_rlnMicrographDefectFile": str,
        "_rlnMicrographDoseRate": float,
        "_rlnMicrographEndFrame": int,
        "_rlnMicrographFrameNumber": int,
        "_rlnMicrographGainName": str,
        "_rlnMicrographId": int,
        "_rlnMicrographMetadata": str,
        "_rlnMicrographMovieName": str,
        "_rlnMicrographName": str,
        "_rlnMicrographNameNoDW": str,
        "_rlnMicrographOriginalPixelSize": float,
        "_rlnMicrographPixelSize": float,
        "_rlnMicrographPreExposure": float,
        "_rlnMicrographShiftX": float,
        "_rlnMicrographShiftY": float,
        "_rlnMicrographStartFrame": int,
        "_rlnMicrographTiltAngle": float,
        "_rlnMicrographTiltAxisDirection": float,
        "_rlnMicrographTiltAxisOutOfPlane": float,
        "_rlnMinimumValue": float,
        "_rlnMinRadiusNnInterpolation": int,
        "_rlnModelStarFile": str,
        "_rlnModelStarFile2": str,
        "_rlnMolecularWeight": float,
        "_rlnMotionModelCoeff": float,
        "_rlnMotionModelCoeffsIdx": int,
        "_rlnMotionModelVersion": int,
        "_rlnMovieFrameNumber": int,
        "_rlnMovieFramesRunningAverage": int,
        "_rlnMtfFileName": str,
        "_rlnMtfValue": float,
        "_rlnNormCorrection": float,
        "_rlnNormCorrectionAverage": float,
        "_rlnNrBodies": int,
        "_rlnNrClasses": int,
        "_rlnNrGroups": int,
        "_rlnNrHelicalAsymUnits": int,
        "_rlnNrHelicalNStart": int,
        "_rlnNrOfFrames": int,
        "_rlnNrOfSignificantSamples": int,
        "_rlnNumberOfIterations": int,
        "_rlnNumberOfIterWithoutChangingAssignments": int,
        "_rlnNumberOfIterWithoutResolutionGain": int,
        "_rlnOffsetRange": float,
        "_rlnOffsetRangeOriginal": float,
        "_rlnOffsetStep": float,
        "_rlnOffsetStepOriginal": float,
        "_rlnOpticsGroup": int,
        "_rlnOpticsGroupName": str,
        "_rlnOpticsStarFile": str,
        "_rlnOrientationalPriorMode": int,
        "_rlnOrientationDistribution": float,
        "_rlnOrientationsID": int,
        "_rlnOrientSamplingStarFile": str,
        "_rlnOriginalImageSize": int,
        "_rlnOriginalParticleName": str,
        "_rlnOriginX": float,
        "_rlnOriginXAngst": float,
        "_rlnOriginXPrior": float,
        "_rlnOriginXPriorAngst": float,
        "_rlnOriginY": float,
        "_rlnOriginYAngst": float,
        "_rlnOriginYPrior": float,
        "_rlnOriginYPriorAngst": float,
        "_rlnOriginZ": float,
        "_rlnOriginZAngst": float,
        "_rlnOriginZPrior": float,
        "_rlnOriginZPriorAngst": float,
        "_rlnOutputRootName": str,
        "_rlnOverallAccuracyRotations": float,
        "_rlnOverallAccuracyTranslations": float,
        "_rlnOverallAccuracyTranslationsAngst": float,
        "_rlnOverallFourierCompleteness": float,
        "_rlnPaddingFactor": float,
        "_rlnParticleBoxFractionMolecularWeight": float,
        "_rlnParticleBoxFractionSolventMask": float,
        "_rlnParticleDiameter": float,
        "_rlnParticleFigureOfMerit": float,
        "_rlnParticleId": int,
        "_rlnParticleName": str,
        "_rlnParticleNumber": int,
        "_rlnParticleSelectZScore": float,
        "_rlnPerFrameCumulativeWeight": float,
        "_rlnPerFrameRelativeWeight": float,
        "_rlnPhaseShift": float,
        "_rlnPipeLineEdgeFromNode": str,
        "_rlnPipeLineEdgeProcess": str,
        "_rlnPipeLineEdgeToNode": str,
        "_rlnPipeLineJobCounter": int,
        "_rlnPipeLineNodeName": str,
        "_rlnPipeLineNodeType": int,
        "_rlnPipeLineProcessAlias": str,
        "_rlnPipeLineProcessName": str,
        "_rlnPipeLineProcessStatus": int,
        "_rlnPipeLineProcessType": int,
        "_rlnPixelSize": float,
        "_rlnPsiStep": float,
        "_rlnPsiStepOriginal": float,
        "_rlnRadiusMaskExpImages": int,
        "_rlnRadiusMaskMap": int,
        "_rlnRandomiseFrom": float,
        "_rlnRandomSeed": int,
        "_rlnRandomSubset": int,
        "_rlnReconstructImageName": str,
        "_rlnReferenceDimensionality": int,
        "_rlnReferenceImage": str,
        "_rlnReferenceSigma2": float,
        "_rlnReferenceSpectralPower": float,
        "_rlnReferenceTau2": float,
        "_rlnRefsAreCtfCorrected": bool,
        "_rlnResolution": float,
        "_rlnResolutionInversePixel": float,
        "_rlnResolutionSquared": float,
        "_rlnSamplingPerturbFactor": float,
        "_rlnSamplingPerturbInstance": float,
        "_rlnSamplingRate": float,
        "_rlnSamplingRateX": float,
        "_rlnSamplingRateY": float,
        "_rlnSamplingRateZ": float,
        "_rlnScheduleBooleanVariableName": str,
        "_rlnScheduleBooleanVariableResetValue": bool,
        "_rlnScheduleBooleanVariableValue": bool,
        "_rlnScheduleCurrentNodeName": str,
        "_rlnScheduleEdgeBooleanVariable": str,
        "_rlnScheduleEdgeInputNodeName": str,
        "_rlnScheduleEdgeIsFork": bool,
        "_rlnScheduleEdgeNumber": int,
        "_rlnScheduleEdgeOutputNodeName": str,
        "_rlnScheduleEdgeOutputNodeNameIfTrue": str,
        "_rlnScheduleEmailAddress": str,
        "_rlnScheduleFloatVariableName": str,
        "_rlnScheduleFloatVariableResetValue": float,
        "_rlnScheduleFloatVariableValue": float,
        "_rlnScheduleJobHasStarted": bool,
        "_rlnScheduleJobMode": str,
        "_rlnScheduleJobName": str,
        "_rlnScheduleJobNameOriginal": str,
        "_rlnScheduleName": str,
        "_rlnScheduleOperatorInput1": str,
        "_rlnScheduleOperatorInput2": str,
        "_rlnScheduleOperatorName": str,
        "_rlnScheduleOperatorOutput": str,
        "_rlnScheduleOperatorType": str,
        "_rlnScheduleOriginalStartNodeName": str,
        "_rlnScheduleStringVariableName": str,
        "_rlnScheduleStringVariableResetValue": str,
        "_rlnScheduleStringVariableValue": str,
        "_rlnSelected": int,
        "_rlnSgdFinalIterations": int,
        "_rlnSgdFinalResolution": float,
        "_rlnSgdFinalSubsetSize": int,
        "_rlnSGDGradientImage": str,
        "_rlnSgdInBetweenIterations": int,
        "_rlnSgdInitialIterations": int,
        "_rlnSgdInitialResolution": float,
        "_rlnSgdInitialSubsetSize": int,
        "_rlnSgdMaxSubsets": int,
        "_rlnSgdMuFactor": float,
        "_rlnSgdSigma2FudgeHalflife": int,
        "_rlnSgdSigma2FudgeInitial": float,
        "_rlnSgdSkipAnneal": bool,
        "_rlnSgdStepsize": float,
        "_rlnSgdSubsetSize": int,
        "_rlnSgdWriteEverySubset": int,
        "_rlnSigma2Noise": float,
        "_rlnSigmaOffsets": float,
        "_rlnSigmaOffsetsAngst": float,
        "_rlnSigmaPriorPsiAngle": float,
        "_rlnSigmaPriorRotAngle": float,
        "_rlnSigmaPriorTiltAngle": float,
        "_rlnSignalToNoiseRatio": float,
        "_rlnSkewnessValue": float,
        "_rlnSmallestChangesClasses": int,
        "_rlnSmallestChangesOffsets": float,
        "_rlnSmallestChangesOrientations": float,
        "_rlnSolventMask2Name": str,
        "_rlnSolventMaskName": str,
        "_rlnSortedIndex": int,
        "_rlnSpectralIndex": int,
        "_rlnSpectralOrientabilityContribution": float,
        "_rlnSphericalAberration": float,
        "_rlnSsnrMap": float,
        "_rlnStandardDeviationValue": float,
        "_rlnStarFileMovieParticles": str,
        "_rlnSymmetryGroup": str,
        "_rlnTau2FudgeFactor": float,
        "_rlnTauSpectrumName": str,
        "_rlnTiltAngleLimit": float,
        "_rlnTransversalDisplacement": float,
        "_rlnUnfilteredMapHalf1": str,
        "_rlnUnfilteredMapHalf2": str,
        "_rlnUnknownLabel": str,
        "_rlnUseTooCoarseSampling": bool,
        "_rlnVoltage": float,
        "_rlnWidthMaskEdge": int,
    }
    if addLabel != False & labelType != False:
        labelList[addLabel] = labelType
    starLabels = starData.columns
    for label in starLabels:
        if label in labelList:
            starData[label] = _get_columns(starData, label, labelList[label])
    return starData


# Input is path to star file, and outputs a pandas dataframe with the information in the starFile
def _open_star(star_file, mode='old'):
    (metaLabels, metaInformation, labels, data) = _get_header(star_file, mode)
    dataList = []
    for entrySeries in data:
        if entrySeries != '':
            entry = [x for x in entrySeries.split() if x != '']
            if entry != []:
                dataList.append(entry)
    starData = pd.DataFrame(data=dataList, columns=labels)
    if metaLabels != []:
        starData.attrs = {
            'metaLabels': metaLabels,
            'metaInformation': metaInformation,
        }
    return _cleanColumns(starData)


# Checks whether starData is PD format - else opens the path and check if it is now PD
def _starDataType(starData, mode='old'):
    if isinstance(starData, str):
        starData = _open_star(starData, mode)
    if isinstance(starData, type(pd.DataFrame())):
        return starData
    else:
        print('Star data is not a DataFrame - exiting')
        return False


def _add_col(starData, value, metaLabel):  # add column to star file
    if type(value) != str and type(value) != float and type(value) != int and type(value) != bool:
        if len(value) == 1:
            starData[metaLabel] = np.repeat(value, len(starData))
        elif len(value) == len(starData):
            starData[metaLabel] = value
        else:
            print(
                'problen with input values - do not match the length of the current dataFrame')
    else:
        starData[metaLabel] = np.repeat(value, len(starData))
        starData[metaLabel] = _get_columns(starData, metaLabel, type(value))
    return starData


def _remove_col(starData, metaLabel):  # Remove column from star file
    return starData.drop(columns=[metaLabel])


def _writeStar(starData, outputFile='./output/output.star', mode='old'):  # Writes out a star file
    '''
    starData outputFile mode
    Used to write out a Star file

    starData is the path to your star file
    outputFile is the path to the output file
    mode tells whether it is pre 3.1 RELION type (old) or post RELION 3.1 style. For now old is default
    '''
    starData = _starDataType(starData, mode)
    if mode == 'new':
        _writeStarNew(starData, outputFile, mode)
        return 'Done'
    output = open(outputFile, 'w')
    output.write('\ndata_\n\nloop_\n')
    c = 1
    for column in starData.columns:
        output.write(f'{column} #{c}\n')
        c += 1
    output.close()
    starData.to_csv(outputFile, sep='\t', header=False, mode='a', index=False)


def _writeStarNew(starData, outputFile='./output/output.star', mode='new'):
    if mode == 'old':
        _writeStar(starData, outputFile, mode)
        return 'done'
    output = open(outputFile, 'w')
    output.write('\ndata_optics\n\nloop_\n')
    c = 1
    for metaLabel in starData.attrs['metaLabels']:
        output.write(f'{metaLabel} #{c}\n')
        c += 1

    for opticGroup in starData.attrs['metaInformation']:
        for entry in opticGroup:
            output.write(f'{entry}\t')
    output.write('\ndata_particles\n\nloop_\n')
    c = 1
    for column in starData.columns:
        output.write(f'{column} #{c}\n')
        c += 1
    output.close()
    starData.to_csv(outputFile, sep='\t', header=False, mode='a', index=False)


def _modCol(starData, metaLabel='_rlnUnknownLabel', value=1, operator='multiplication'):
    if not metaLabel.startswith('_rln', 0, 4):
        metaLabel = f'_rln{metaLabel}'
    if operator == 'addition' or operator == '+':
        starData[metaLabel] = _get_columns(
            starData, metaLabel, convertType=float)+value
        return starData
    elif operator == 'subtraction' or operator == '-':
        starData[metaLabel] = _get_columns(
            starData, metaLabel, convertType=float)-value
        return starData
    elif operator == 'division' or operator == '/':
        value = 1/value
    if operator == 'multiplication' or operator == 'division' or operator == '*' or operator == '/':
        starData[metaLabel] = _get_columns(
            starData, metaLabel, convertType=float)*value
        return starData
    else:
        print(
            f'Operator {operator} is unknown - Please try again with a proper operator (division/multiplication/addition/subtraction)')


def modify_column(starData, metaLabel='_rlnUnknownLabel', value=1, operator='multiplication', outputFile='./output/output.star', mode='old'):
    '''
    starData metaLabel value operator outputFile mode --- Used to modify number columns

    starData is the path to your star file
    metaLabel is the column you want to change (defailt = _rlnUnknownLabel)
    value is the value by which you want to change it (default = 1)
    operator is which operator to used (Default = multiplication) Supported operators are: multiplication (*), division (/), addition (+), and subtraction (-).
    outputFile is the path to the output star file. Defaults to ./output/output.star
    mode is whether it is a pre RELION 3.1 ('old') or post RELION 3.1 ('new') starFile. Defaults to 'old'
    '''
    starData = _starDataType(starData, mode)
    starData = _modCol(starData, metaLabel, value, operator)
    _writeStar(starData, outputFile, mode)

# Mode new not implemented


def bin_star(starData, bin=1, outputFile='./output/output.star', mode='old'):
    starData = _starDataType(starData, mode)
    if mode == 'old':
        labelList = ['_rlnCoordinateX', '_rlnCoordinateY',
                     '_rlnCoordinateZ', '_rlnOriginX', '_rlnOriginY', '_rlnOriginZ']
        pixelSizeList = ['_rlnPixelSize',
                         '_rlnDetectorPixelSize', '_rlnImagePixelSize']
        allLabels = starData.columns
        for label in labelList:
            if label in allLabels:
                starData = _modCol(starData, label, bin, 'division')
        for pixelLabel in pixelSizeList:
            if pixelLabel in labelList:
                starData = _modCol(starData, label, bin, 'multiplication')
    if mode == 'new':
        print('This is not implemented for the new relion format yet')
        pass
    return starData


def convertStarToNew(starData, mode='old', imageSize=100, tomo=True, Cs=2.7, voltage=300, pixelSize=False):
    '''
    starData mode imageSize tomo Cs voltage ---
    Used to convert star file to new format (post RELION 3.1)

    starData is the path to your star file
    mode is whether you want to convert from old to new format (old) or from new to old formate (new)
    imageSize is the size of your particles (in pixels)
    tomo (True/False, default = True) is whether the images are tomography images (3 dimensional) or 2D images
    Cs is the spherical abberation (in mm, default = 2.7 mm)
    voltage is the acceleration of the electrons by the microscope (in kV, default = 300)
    pixelSize is the pixelsize (in A, default = False, meaning the pixelSize is already in the starFile)
    '''
    if mode == 'new':
        return convertStarToOld(starData, mode, tomo)
    starData = _starDataType(starData, mode)
    starData = _cleanColumns(starData)
    if tomo == True:
        imgDim = 3
    else:
        imgDim = 2
    oriLabels = starData.columns
    metaLabels = ['_rlnOpticsGroup', '_rlnOpticsGroupName', '_rlnImageSize',
                  '_rlnImageDimensionality', '_rlnVoltage', '_rlnSphericalAberration']
    metaInformation = [1, 'opticsGroup1', imageSize, imgDim, voltage, Cs]
    if pixelSize != False:
        metaLabels.append('_rlnPixelSize')
        metaInformation.append(pixelSize)
    else:
        pixelSizeLabels = ['_rlnDetectorPixelSize',
                           '_rlnImagePixelSize',
                           '_rlnPixelSize'
                           ]
        pBool = False
        for plabel in pixelSizeLabels:
            if plabel in oriLabels:
                pixelSize = starData[plabel][0]
                pPrintLabel = plabel
                pBool = True
                starData = _remove_col(starData, plabel)
        if pBool == False:
            print(f'No pixel size was found')
        else:
            metaLabels.append('_rlnImagePixelSize')
            metaInformation.append(pixelSize)
    starData.attrs = {
        'metaLabels': metaLabels,
        'metaInformation': [metaInformation],
    }
    starData = _add_col(starData, 1, '_rlnOpticsGroup')
    return starData

# Still needs to be tested


def convertStarToOld(starData, mode='new', tomo=True, pixelSize=False):
    '''
    starData mode imageSize tomo ---
    Used to convert a star file from post RELION 3.1 format to old format

    starData is the path to your star file
    mode is whether you want to convert from old to new format (old) or from new to old formate (new)
    imageSize is the size of your particles (in pixels)
    tomo (True/False, default=True) is whether the images are tomography images (3 dimensional) or 2D images
    '''
    if mode == 'old':
        return convertStarToNew(starData, mode, tomo)
    starData = _starDataType(starData, mode)
    if len(starData.attrs['metaInformation']) > 1:
        # Not implemented yet...
        print('Your star file has more than one optics group. Please use the split_Optics_files instead before converting')
        return False
    metaInformation = starData.attrs['metaInformation'][0]
    metaLabels = starData.attrs['metaLabels']
    print(metaLabels)
    metaDict = {}
    for i in range(len(metaLabels)):
        metaDict[metaLabels[i]] = metaInformation[i]
    labelList = ['_rlnDetectorPixelSize',
                 '_rlnPixelSize', "_rlnImagePixelSize"]
    for label in labelList:
        if label in metaDict:
            starData = _add_col(starData, metaDict[label], label)
            if pixelSize == False:
                pixelSize = metaDict[label]
                print(
                    f'pixel size of {pixelSize} is being used to fix origin, extracted from {label} \n If this is not correct please specify the pixel size manually')
    if pixelSize != False:
        starData = _fix_origins(starData, pixelSize=pixelSize)
    removeLabelList = ['_rlnOpticsGroup']
    for label in removeLabelList:
        starData = _remove_col(starData, label)
    return starData

# Origin fix


def _fix_origins(starData, pixelSize, originLabels=['_rlnOriginX', '_rlnOriginY', '_rlnOriginZ']):
    for originLabel in originLabels:
        if f'{originLabel}Angst' in starData.columns:
            starData = _modCol(
                starData, f'{originLabel}Angst', float(pixelSize), 'division')
            starData = starData.rename(
                columns={f'{originLabel}Angst': originLabel})
    return starData

# Reset offsets
def reset_offsets(star_data: pd.DataFrame, mode: str, pixel_size: float =False) -> pd.DataFrame:
    if mode == 'new':
        star_data = _fix_origins(star_data,pixel_size)
    for coord in 'XYZ':
        if f'_rlnOrigin{coord}' in star_data:
            star_data[f'_rlnCoordinate{coord}'] = star_data[f'_rlnCoordinate{coord}'] - star_data[f'_rlnOrigin{coord}']
    return star_data

def convertStarToPySEGold(starData, mode='new', tomo=True, pixelSize=6.802):
    '''
    starData mode imageSize tomo ---
    Used to convert a star file from post RELION 3.1 format to old format

    starData is the path to your star file
    mode is whether you want to convert from old to new format (old) or from new to old formate (new)
    imageSize is the size of your particles (in pixels)
    tomo (True/False, default=True) is whether the images are tomography images (3 dimensional) or 2D images
    '''
    if mode == 'old':
        return convertStarToNew(starData, mode, tomo)
    starData = _starDataType(starData, mode)
    starData = _fix_origins(starData, pixelSize)
    removeLabelList = ['_rlnOpticsGroup',
                       '_rlnCtfMaxResolution', '_rlnPixelSize']
    for label in removeLabelList:
        starData = _remove_col(starData, label)
    return starData


def add_column(starData, outputFile='output/output.star', value=0, metaLabel='_rlnUnknownLabel', mode='old'):
    '''
    starData, outputFile, value, metaLabel, mode ---
    Used to add a new column to the star file

    starData is the path to your star file
    outputFile is the path to the output file
    value is the value to be added to the star file (default 0)
    metaLabel is the label you want to add to the starFile (default _rlnUnknownLabel)
    '''
    if not metaLabel.startswith('_rln', 0, 4):
        metaLabel = f'_rln{metaLabel}'
    starData = _starDataType(starData, mode)
    starData = _add_col(starData, value, metaLabel)
    _writeStar(starData, outputFile, mode)


def remove_column(starData, outputFile='output/output.star', metaLabel='_rlnUnknownLabel', mode='old'):
    '''
    starData, outputFile, metaLabel, mode ---
    Used to remove a column from the star file

    starData is the path to your star file
    outputFile is the path to the output file
    metaLabel is the label you want to remove to the starFile (default _rlnUnknownLabel)
    '''
    starData = _starDataType(starData, mode)
    if type(metaLabel) == str:
        if not metaLabel.startswith('_rln', 0, 4):
            metaLabel = f'_rln{metaLabel}'
        starData = _remove_col(starData, metaLabel)
    elif type(metaLabel) == (list):
        for entry in metaLabel:
            if not entry.startswith('_rln', 0, 4):
                metaLabel = f'_rln{metaLabel}'
            starData = _remove_col(starData, entry)
    _writeStar(starData, outputFile, mode)


def combine_star_files(ListofStarData, mode='old', referenceDataIndex=0):
    print(
        f'Reference star file which decideds columns is {ListofStarData[referenceDataIndex]}')
    if mode == 'old':
        return pd.concat([x[ListofStarData[referenceDataIndex].columns] for x in ListofStarData])
    if mode == 'new':
        res_star = pd.concat(
            [x[ListofStarData[referenceDataIndex].columns] for x in ListofStarData])
        res_star.attrs = ListofStarData[referenceDataIndex].attrs
        return res_star


def combine_star_files_new(ListofStarData, mode='new', referenceDataIndex=0):
    print(
        f'Reference star file which decideds columns is {ListofStarData[referenceDataIndex]}')
    if mode != 'new':
        return combine_star_files(ListofStarData, mode=mode)
    combined_star = pd.concat([x[ListofStarData[referenceDataIndex].columns]
                              for x in ListofStarData], ignore_index=True)
    combined_star.attrs = ListofStarData[referenceDataIndex].attrs
    return combined_star


if __name__ == '__main__':
    starData = f'/g/scb/mahamid/rasmus/edmp/hep_ref.star'
    output = f'/g/scb/mahamid/rasmus/edmp/hep_ref_n.star'
    _writeStar(star, output, mode='old')