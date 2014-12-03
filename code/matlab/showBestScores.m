function showBestScores(data)

bestScores = zeros(1,length(data));
for k=1:length(data)
  scores = zeros(1,length(data(k).patches));
  
  for i=1:length(data(k).patches)
    patch = data(k).patches(i).patch;

  %patch(rangeLY,rangeLX) = patch(rangeRY,rangeRX);


    scores(i) = zssd(data(k).refPatch, patch);
  end
  bestScores(k) = min(scores);
end
hist(bestScores);