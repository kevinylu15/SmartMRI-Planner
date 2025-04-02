// SmartMRI Planner - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const mriForm = document.getElementById('mriForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultContainer = document.getElementById('resultContainer');
    
    mriForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultContainer.style.display = 'none';
        
        // Get form data
        const formData = new FormData(mriForm);
        
        // Add files from file input
        const fileInput = document.getElementById('paperUpload');
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append('papers', fileInput.files[i]);
        }
        
        // Add paper URLs
        formData.append('paper_urls', document.getElementById('paperUrl').value);
        
        // Add patient info
        formData.append('patient_info', document.getElementById('patientInfo').value);
        
        // Use the actual processing endpoint
        fetch('/api/process', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Hide loading indicator
                loadingIndicator.style.display = 'none';
                
                // Display results
                displayResults(data);
            })
            .catch(error => {
                console.error('Error:', error);
                loadingIndicator.style.display = 'none';
                alert('An error occurred while processing your request. Please try again.');
            });
    });
    
    function displayResults(recommendation) {
        // Populate the results
        document.getElementById('fieldStrength').textContent = recommendation.field_strength;
        document.getElementById('contrastAgent').textContent = recommendation.contrast_agent;
        document.getElementById('rationale').textContent = recommendation.rationale;
        
        // Populate sequences
        const sequencesList = document.getElementById('sequences');
        sequencesList.innerHTML = '';
        recommendation.sequences.forEach(sequence => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = sequence;
            sequencesList.appendChild(li);
        });
        
        // Populate special considerations
        const considerationsList = document.getElementById('specialConsiderations');
        considerationsList.innerHTML = '';
        recommendation.special_considerations.forEach(consideration => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = consideration;
            considerationsList.appendChild(li);
        });
        
        // Populate contraindications
        const contraindicationsList = document.getElementById('contraindications');
        contraindicationsList.innerHTML = '';
        recommendation.contraindications.forEach(contraindication => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = contraindication;
            contraindicationsList.appendChild(li);
        });
        
        // Populate alternative options
        const alternativesContainer = document.getElementById('alternativeOptions');
        alternativesContainer.innerHTML = '';
        recommendation.alternative_options.forEach(option => {
            const card = document.createElement('div');
            card.className = 'card alternative-card';
            
            const cardBody = document.createElement('div');
            cardBody.className = 'card-body';
            
            const title = document.createElement('h5');
            title.className = 'card-title';
            title.textContent = `Alternative: ${option.field_strength}`;
            
            const sequencesTitle = document.createElement('h6');
            sequencesTitle.textContent = 'Sequences:';
            
            const sequencesList = document.createElement('ul');
            sequencesList.className = 'list-group list-group-flush mb-3';
            option.sequences.forEach(sequence => {
                const li = document.createElement('li');
                li.className = 'list-group-item';
                li.textContent = sequence;
                sequencesList.appendChild(li);
            });
            
            const rationale = document.createElement('p');
            rationale.className = 'card-text';
            rationale.textContent = option.rationale;
            
            cardBody.appendChild(title);
            cardBody.appendChild(sequencesTitle);
            cardBody.appendChild(sequencesList);
            cardBody.appendChild(document.createElement('h6')).textContent = 'Rationale:';
            cardBody.appendChild(rationale);
            
            card.appendChild(cardBody);
            alternativesContainer.appendChild(card);
        });
        
        // Populate analyzed papers
        const papersList = document.getElementById('analyzedPapers');
        papersList.innerHTML = '';
        recommendation.analyzed_papers.forEach(paper => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = paper;
            papersList.appendChild(li);
        });
        
        // Show the results container
        resultContainer.style.display = 'block';
    }
});
