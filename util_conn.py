import jax.random as rng

def make_V1maps(Lxy, λ_map, gridperdeg, PLOT=False, key_id=0):
    """
    make retinotopic and orientation maps for V1
    Lxy: x-edge and y-edge lengths of cortical region in degrees of visual angle
    λ_map: hypercolumn size (= period of orientation map)  in degrees of visual angle
    gridperdeg: grid points per degree of visual angle
    key_id: key id for the JAX PRNG
    """
    gridsize = 1 + np.round(Lxy * gridperdeg)
    X, Y = np.meshgrid(np.linspace(0,Lxy[0], gridsize[0]), np.linspace(0,Lxy[1], gridsize[1]))

    kc=(2*np.pi)/λ_map
    z = np.zeros(X.shape)
    n=30  # number of plane waves in orientation map
    key = rng.PRNGKey(key_id)
    for j in range(n):
        key, gkey = rng.split(key)
        kj = kc * np.array([np.cos(j*np.i/n), np.sin(j*np.pi/n)])
        sj = 2* rng.bernoulli(gkey) - 1
        key, gkey = rng.split(key)
        phij = rng.uniform(gkey)*2*np.pi
        tmp = (X*kj[0] + Y*kj[1])*sj + phij
        z = z + np.exp(1j * tmp)
     ori_map = np.angle(z)
     ori_map =  ori_map -  ori_map.min()
     ori_map =  ori_map*180/(2*np.pi)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(10001)
        plt.imshow( ori_map, cmap="hsv")
        plt.colorbar()

    return X, Y, ori_map, gridsize

def FullModelBuild_UniformStrength_modifiedbyYashar_normalized(ssn, PLOT=False):
    """
    Modifications by Yashar Ahmadian: I removed periodic boundary conditions in
    spatial connectivity (search "periodic")
     Things are normalized such that ssn.J0's xy-element gives the total sum of weights (in the particular
     realization of NxN matrix ssn.JW constructed) received by every neuron of type x from neurons of type y.
     If ssn.CELLWISENORMALIZED =1, the above is literally true: i.e. total sum of each type of weight received by ALL neurons of a certain type are equal to each other (and to corresponding element of ssn.J0)
     If ssn.CELLWISENORMALIZED =0, (more realistic) then "received by every neuron" must be replaced by "received by the
     average neuron", where average means empricial average in the particular realization of ssn.JW, and the latter's block-row sums are not literally equal, but average to ssn.J0(x,y).

     Difference with FullModelBuild_UniformStrength_modifiedbyYashar.m are
     denoted by Diff#?? below in comments

    # "Full Model" - Orientation and Spatially-dependent connections
    # Try sparseness a different way -- all connections of a given type are
    # the same strength, but now the probability of connection decreases with
    # distance in space and orientation
    """

    # Diff#1: no real change, next line used to be: λ_map = ssn.λ_map/ssn.MagnFactor  now this conversion is done in file calling this one (makeSSNconnectivity.m)
    X, Y, ori_map, gridsize = make_V1maps(ssn.Lxy, ssn.lambda_map, gridperdeg, PLOT=PLOT)
    Len =  ori_map.shape
    ssn.gridsize = gridsize
    ssn.dxy = ssn.Lxy/(ssn.gridsize-1) # dx and dy (x and y grid spacing) in degrees
    ssn.X = X.ravel() - ssn.Lxy[0]/2 # in deg
    ssn.Y = Y.ravel() - ssn.Lxy[1]/2 # in deg
    ssn.ori_map =  ori_map.ravel()
    ssn.Ne = ssn.ori_map.size # prod(size(ssn. ori_map))
    ssn.topos_vec = np.vstack([ssn.X, ssn.Y, ssn.ori_map]) # shape: 3 x ssn.Ne



    [X Y] = ind2sub(size( ori_map),1:numel( ori_map))

    XYZ = [X' Y'  ori_map(:)]



            # #  Network parameters


    JNoise = ssn.JNoise # 0.25
    alphaE = ssn.alpha[0] # 0.1 # \kappa_E in paper
    alphaI = ssn.alpha[1] # 0.5 # \kappa_I in paper

    J0 = ssn.J0
    # Diff#2:            (important --c.f. Diff#6) file used to have the following 2 lines.
    #       The purpose was to make individual synaptic weights scale like 1/K --we now do that by direct normalization by number of connections received --see Diff#6
    #       The factors divided by in these commented lines (up to a constant factor) the total some of probabilities (hence the average # of connections) of each type received by a neuron of each type.
    #       ssn.J0 is the total-weight-received, i.e. that which we would not want to have scale with N. The operation here gives the individual synaptic weights that do scale like 1/K.
    #       Note that the normalization was not perfect (we would not divide by the exact actual number of connections in the random realization, as we do now below --see Diff#6-- or not even the correct sum of the connection-probabilities --but it is only off from the latter by a constant factor (which would depend on Gaussian or Exponential fall off --hence possibly different for E and I projections), and hence it took care of scaling of individual weights by 1/K.
    #                 J0 = ssn.J0./(2*pi*ssn.sigmaXYdeg.^2 * ssn.gridperdeg^2) # ssn.J0 # note that ssn.gridperdeg = 1/ssn.dxy[0] almost
    #                 J0 = J0./repmat(ssn.alpha,2,1)
    Jee =  J0[0,0]
    Jie =  J0[1,0]
    Jei = -J0[0,1]
    Jii = -J0[1,1]

    SigmaEE = ssn.sigmaXYdeg[0,0]
    SigmaIE = ssn.sigmaXYdeg[1,0]
    SigmaEI = ssn.sigmaXYdeg[0,1]
    SigmaII = ssn.sigmaXYdeg[1,1]


                        # # Width of connectivity in orientation domain (in degrees)
                        # Malach et al (1993) observe fairly widespread connectivity
    SigmaOri = ssn.SigmaOri # 45




                        #
    # #  There are two ways to make synapses "sparse" -- the second is to use distance to define probability of connection:

       tic

      szO = size( ori_map)
      N = numel( ori_map)    #  Number of neurons in the network (E and I)


                        # MinSyn = 1e-5
                        MinSyn = 1e-4

                        Xposition = XYZ(:,1)'
                        XdMat = repmat(Xposition,N,1)
                        XDist = abs(bsxfun(@minus,XdMat,XdMat'))

                        Yposition = XYZ(:,2)'
                        YdMat = repmat(Yposition,N,1)
                        YDist = abs(bsxfun(@minus,YdMat,YdMat'))

                        OriPref = XYZ(:,3)'
                        OridMat = repmat(OriPref,N,1)
                        OriDist = abs(bsxfun(@minus,OridMat,OridMat'))

    #                      # Yashar: Dan's imposing periodic boundary conditions
                        if isfield(Jp,'PERIODIC') && ssn.PERIODIC# Yashar added this so in default case no periodic b.c.
                            XDist(XDist > szO(1)/2) = szO(1) - XDist(XDist > szO(1)/2)
                            YDist(YDist > szO(1)/2) = szO(1) - YDist(YDist > szO(1)/2)
                        end

                        OriDist(OriDist > 90) = 180 - OriDist(OriDist > 90)
                        deltaD = sqrt((XDist).^2 + (YDist).^2)
                        deltaD = deltaD*ssn.dxy[0] # convertint to degrees

                        clear YDist XDist XdMat YdMat OridMat

                        if ssn.Exp_Eprofile# Exp distrib has same normalization as Gaussian in 2D , so don't worry about it
                            Wee = exp(-abs(deltaD)/SigmaEE - OriDist.^2/(2*SigmaOri^2))
                            Wie = exp(-abs(deltaD)/SigmaIE - OriDist.^2/(2*SigmaOri^2))
                        else# use Gaussian profile for E projections
                            Wee = exp(-deltaD.^2/(2*SigmaEE^2) - OriDist.^2/(2*SigmaOri^2))
                            Wie = exp(-deltaD.^2/(2*SigmaIE^2) - OriDist.^2/(2*SigmaOri^2))
                        end
                        Wei = exp(-deltaD.^2/(2*SigmaEI^2) - OriDist.^2/(2*SigmaOri^2))
                        Wii = exp(-deltaD.^2/(2*SigmaII^2) - OriDist.^2/(2*SigmaOri^2))

    # In next commented lines I wanted to normalize things such that, when ssn.SPARSE==1, the avg # of
    # connections does not scale with ssn.gridperdeg, but first of all perhaps this is unnecess, band 2nd I'm now more interested ssn.SPARSE==0
    #              M = round(size(Wee,1)/2)
    #              Norm = sum(Wee(M,:))
                Wee = alphaE*Wee
                        Wee(Wee < MinSyn) = 0
                        Wee = sparse(Wee)

                Wie = alphaE*Wie
                        Wie(Wie < MinSyn) = 0
                        Wie = sparse(Wie)

                Wei = alphaI*Wei
                        Wei(Wei < MinSyn) = 0
                        Wei = sparse(Wei)

                Wii = alphaI * Wii
                        Wii(Wii < MinSyn) = 0
                        Wii = sparse(Wii)

                        clear OriDist deltaD
                toc

     # #  Normalize before randomizing:
    #
    #        mWee1 = mean(mean(Wee))
    #        mWie1 = mean(mean(Wie))
    #        mWei1 = mean(mean(Wei))
    #        mWii1 = mean(mean(Wii))
    #  #

    #  #
    #       tWee = mean(Wee,2)
    #       Wee = bsxfun(@times,Wee,(mWee./tWee))
    #
    #       tWie = mean(Wie,2)
    #       Wie = bsxfun(@times,Wie,(mWie./tWie))
    #
    #       tWei = mean(Wei,2)
    #       Wei = bsxfun(@times,Wei,(mWei./tWei))
    #
    #       tWii = mean(Wii,2)
    #       Wii = bsxfun(@times,Wii,(mWii./tWii))
    #
    #
    #        mWeeO = mWee
    #        mWieO = mWie
    #        mWeiO = mWei
    #        mWiiO = mWii

    # #   And sparsify (if SPARSE = 1):
    # ssn.SPARSE =0 # whether (1) to sparsify JW by random-binarizing of weights or (0) to use the probabilities of the that bernouli distrib as the weights themselves (all before adding multiplicative noise given by ssn.JNoise)


    # Diff#3: we now allow for uniform non-sparse noise distribution when ssn.JNoise_Normal == 0
    if ssn.JNoise_Normal
        rand_dist = @(x) sprandn(x) #  normal distibution
    else
        rand_dist = @(x) 2*sprand(x) - 1 # uniform distibution # has std of 1/sqrt(3)
    end

    # Diff#4: we now allow for possibility of non-sparsified connectivity when ssn.SPARSE ==0
    tic
    if ssn.SPARSE
        sWee = sprand(Wee)
        sWee = (Wee > sWee)
    else
        sWee = Wee
    end
    #  Diff#5: (important): instead of next 2 lines we do the following 2, i.e.
    #            we don't multiply by Jee at this point, but further down after normalizations of W's
    #  mWee = mean(mean(Jee.*sWee))
    #  Wee = (Jee + (Jee*JNoise*rand_dist(sWee))).*sWee
    mWee = mean(mean(sWee)) # not used (unlike in FullModelBuild_UniformStrength_modifiedbyYashar)
    Wee = (1 + (JNoise*rand_dist(sWee))).*sWee
    Wee(Wee < 0) = 0
    clear sWee

    if ssn.SPARSE
        sWie = sprand(Wie)
        sWie = (Wie > sWie)
    else
        sWie = Wie
    end
    #  Diff#5: (important): instead of next 2 lines we do the following 2, i.e.
    #            we don't multiply by Jie at this point, but further down after normalizations of W's
    #  mWie = mean(mean(Jie.*sWie))
    #  Wie = (Jie + (Jie*JNoise*rand_dist(sWie))).*sWie
    mWie = mean(mean(sWie)) # not used (unlike in FullModelBuild_UniformStrength_modifiedbyYashar)
    Wie = (1 + (JNoise*rand_dist(sWie))).*sWie
    Wie(Wie < 0) = 0
    clear sWie

    if ssn.SPARSE
        sWei = sprand(Wei)
        sWei = (Wei > sWei)
    else
        sWei = Wei
    end
    #  Diff#5: (important): instead of next 2 lines we do the following 2, i.e.
    #            we don't multiply by Jei at this point, but further down after normalizations of W's
    #  mWei = mean(mean(Jei.*sWei))
    #  Wei = (Jei + (Jei*JNoise*rand_dist(sWei))).*sWei
    mWei = mean(mean(sWei))  # not used (unlike in FullModelBuild_UniformStrength_modifiedbyYashar)
    Wei = (1 + (JNoise*rand_dist(sWei))).*sWei
    Wei(Wei < 0) = 0
    clear sWei

    if ssn.SPARSE
        sWii = sprand(Wii)
        sWii = (Wii > sWii)
    else
        sWii = Wii
    end
    #  Diff#5: (important): instead of next 2 lines we do the following 2, i.e.
    #            we don't multiply by Jii at this point, but further down after normalizations of W's
    #  mWii = mean(mean(Jii.*sWii))
    #  Wii = (Jii + (Jii*JNoise*rand_dist(sWii))).*sWii
    mWii = mean(mean(sWii)) # not used (unlike in FullModelBuild_UniformStrength_modifiedbyYashar)
    Wii = (1 + (JNoise*rand_dist(sWii))).*sWii
    Wii(Wii < 0) = 0
    clear sWii



    ssn.meanW = full([mWee, -mWei  mWie, -mWii])

    toc

     # #  Normalize Wxy's (despite added noise and random sparsification)
    #  Note that the normalization done here (in either case, but unlike that done in Dan's original code)
    #  takes care of the scaling of individual synaptic weights like 1/K


    #  ssn.CELLWISENORMALIZED = 1 #  if 1: Normalize so all row-sums in Wxy are  equal to 1
    #                     #  if 0: Normalize so the average row-sum in Wxy is equal to 1

    # Diff#6: (important): In the case ssn.CELLWISENORMALIZED==1 (Dan's case), we now normalize so that (eventually) the total sum of input
    # weights to every neuron in the XY (X,Y = E,I) quadrant of JW is equal to corresponding element of J0
    # While before the mean input (e.g. tWee) was normalized to the mean input averaged over all neurons (e.g. mWee)s
    #  In the case ssn.CELLWISENORMALIZED==0, normalization is such that the total-sum-of-weights from each quadrant of JW matches the value in
    #  J0 only after averaging over postsynaptic neurons in that quadrant.
    #  Note that the normalization done here (in either case, but unlike that done in Dan's original code)
    #  takes care of the scaling of individual synaptic weights like 1/K
    #  Thus there is no need for the more rough scaling that was done in commented lines in Diff#2 previously.
    #
    #      tWee = mean(Wee,2)
    #      Normal = mWee./tWee
         tWee = sum(Wee,2)
         if ssn.CELLWISENORMALIZED
            Normal = 1./tWee
         else
            Normal = 1/mean(tWee)
         end
         if all(~isnan(Normal) & ~isinf(Normal))
            Wee = bsxfun(@times,Wee,Normal)
         else
             warning('JW not normalized')
         end

    # Diff#6: (important): see above
    #      tWie = mean(Wie,2)
    #      Normal = mWie./tWie
         tWie = sum(Wie,2)
         if ssn.CELLWISENORMALIZED
            Normal = 1./tWie
         else
            Normal = 1/mean(tWie)
         end
         if all(~isnan(Normal) & ~isinf(Normal))
             Wie = bsxfun(@times,Wie,Normal)
         else
             warning('JW not normalized')
         end

    # Diff#6: (important): see above
    #      tWei = mean(Wei,2)
    #      Normal = mWei./tWei
         tWei = sum(Wei,2)
         if ssn.CELLWISENORMALIZED
            Normal = 1./tWei
         else
            Normal = 1/mean(tWei)
         end
         if all(~isnan(Normal) & ~isinf(Normal))
             Wei = bsxfun(@times,Wei,Normal)
         else
             warning('JW not normalized')
         end

    # Diff#6: (important): see above
    #      tWii = mean(Wii,2)
    #      Normal = mWii./tWii
         tWii = sum(Wii,2)
         if ssn.CELLWISENORMALIZED
            Normal = 1./tWii
         else
            Normal = 1/mean(tWii)
         end
         if all(~isnan(Normal) & ~isinf(Normal))
             Wii = bsxfun(@times,Wii,Normal)
         else
             warning('JW not normalized')
         end


          toc

    # Diff#7: (important): add a local (delta-function) connectivity component
    Wee = ssn.Plocl[0,0] * speye(size(Wee))  + (1-ssn.Plocl[0,0]) * Wee
    Wie = ssn.Plocl[1,0] * speye(size(Wie))  + (1-ssn.Plocl[1,0]) * Wie
    Wei = ssn.Plocl[0,1] * speye(size(Wei))  + (1-ssn.Plocl[0,1]) * Wei
    Wii = ssn.Plocl[1,1] * speye(size(Wii))  + (1-ssn.Plocl[1,1]) * Wii
    # note that Wxy still remain normalized

    # Diff#8: see Diff#5 above
    # JW = [Wee, -Wei  Wie , -Wii]  # changed this to next line
    JW = [Jee*Wee, -Jei*Wei  Jie*Wie , -Jii*Wii]


    # #  Visualize E/I projections of one grid point

    # Yashar: I commented this

    # cellN = randi(N,1)
    cellN = sub2ind(szO,round(szO(1)/2),round(szO(1)/2)) # round((N+1)/2)
    # cellN = round(szO(1)/2)

    eee = .000
    WeeFromMid = Wee(:,cellN)
    WeeFromMid(WeeFromMid< eee*max(WeeFromMid)) = 0
    WeeFM = reshape(WeeFromMid, Len[0], Len[1])


    WieFromMid = Wie(:,cellN)
    WieFromMid(WieFromMid< eee*max(WieFromMid)) = 0
    WieFM = reshape(WieFromMid, Len[0], Len[1])


    WeiFromMid = Wei(:,cellN)
    WeiFromMid(WeiFromMid< eee*max(WeiFromMid)) = 0
    WeiFM = reshape(WeiFromMid, Len[0], Len[1])


    WiiFromMid = Wii(:,cellN)
    WiiFromMid(WiiFromMid< eee*max(WiiFromMid)) = 0
    WiiFM = reshape(WiiFromMid, Len[0], Len[1])


    clipping = @sign
    clipping = @(x) x
    figure(1999)
    subplot(2,2,1)
    imagesc(clipping(WeeFM))
    xlabel('E to E projections')


    subplot(2,2,2)
    imagesc(clipping(WeiFM))
    xlabel('I to E projections')

    subplot(2,2,3)
    imagesc(clipping(WieFM))
    xlabel('E to I projections')


    subplot(2,2,4)
    imagesc(clipping(WiiFM))
    xlabel('I to I projections')











    # #  nonlinearity parameters:   # not exported and not used # relilc of Dan's code
                        n = 2.0
                        k = 0.01

                        # # Time constants

                        tau_E = 20            #  Excitatory time constant
                        tau_I = 10            #  Inhibitory time constant

                        # # More variability: cellular parameters
                        NLnoise = 0.05

    #                     JNoise = 0.1
    #                     NLnoise = 0.05

                        # # Add variance to parameters

                        kE = (k + (k*NLnoise)*randn(N,1))
                        kI = (k + (k*NLnoise)*randn(N,1))

                        nE = (n + (n*NLnoise)*randn(N,1))
                        nI = (n + (n*NLnoise)*randn(N,1))

                        tauE = (tau_E + (tau_E*NLnoise)*randn(N,1))
                        tauI = (tau_I + (tau_I*NLnoise)*randn(N,1))

                        # # And restore original means
                        tauE = tauE + (tau_E - mean(tauE))
                        tauI = tauI + (tau_I - mean(tauI))

                        kE = kE + (k - mean(kE))
                        kI = kI + (k - mean(kI))

                        nE = nE + (n - mean(nE))
                        nI = nI + (n - mean(nI))


               # #
    #
    #  	# sketch to compare gaussian std with Bosking et al. 1997 pictures
    #
    #  N = 1000
    #
    #  x = randn(N,1)
    #  y = randn(N,1)
    #
    #  thet = linspace(0,2*pi,200)
    #  figure(3) clf
    #  hold on
    #  plot(cos(thet),sin(thet),'k')
    #  plot(x,y,'.','markersize',10)
    #  hold off
    #  axis equal
    #
    # save
