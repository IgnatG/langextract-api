// LangExtract API — Go client example.
//
// Demonstrates:
//   1. Submit an extraction from raw text.
//   2. Submit an extraction from a URL.
//   3. Submit a batch.
//   4. Poll a task until it completes.
//
// Uses only the Go standard library — no external dependencies.
//
// Usage:
//   go run examples/go/client.go

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const (
	defaultAPIBase     = "http://localhost:8000/api/v1"
	defaultProvider    = "gpt-4o"
	pollIntervalSec    = 2
	pollTimeoutSec     = 120
)

func apiBase() string {
	if v := os.Getenv("API_BASE"); v != "" {
		return v
	}
	return defaultAPIBase
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// ExtractionConfig holds optional LangExtract pipeline overrides.
type ExtractionConfig struct {
	PromptDescription string  `json:"prompt_description,omitempty"`
	Temperature       float64 `json:"temperature,omitempty"`
}

// ExtractionRequest is the body for POST /extract.
type ExtractionRequest struct {
	RawText          string           `json:"raw_text,omitempty"`
	DocumentURL      string           `json:"document_url,omitempty"`
	Provider         string           `json:"provider,omitempty"`
	Passes           int              `json:"passes,omitempty"`
	IdempotencyKey   string           `json:"idempotency_key,omitempty"`
	ExtractionConfig ExtractionConfig `json:"extraction_config,omitempty"`
}

// SubmitResponse is the body returned by POST /extract.
type SubmitResponse struct {
	TaskID  string `json:"task_id"`
	Status  string `json:"status"`
	Message string `json:"message"`
}

// BatchRequest is the body for POST /extract/batch.
type BatchRequest struct {
	BatchID   string              `json:"batch_id"`
	Documents []ExtractionRequest `json:"documents"`
	Provider  string              `json:"provider,omitempty"`
}

// BatchSubmitResponse is the body returned by POST /extract/batch.
type BatchSubmitResponse struct {
	BatchID     string   `json:"batch_id"`
	TaskIDs     []string `json:"task_ids"`
	BatchTaskID string   `json:"batch_task_id"`
}

// Entity is a single extracted entity in a completed task result.
type Entity struct {
	ExtractionClass string            `json:"extraction_class"`
	ExtractionText  string            `json:"extraction_text"`
	Attributes      map[string]string `json:"attributes,omitempty"`
}

// TaskResult holds the list of extracted entities.
type TaskResult struct {
	Entities []Entity `json:"entities"`
}

// TaskResponse is the body returned by GET /tasks/{id}.
type TaskResponse struct {
	TaskID string      `json:"task_id"`
	State  string      `json:"state"`
	Result *TaskResult `json:"result,omitempty"`
	Error  string      `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

// postJSON marshals payload, POSTs to url, and decodes the response into dst.
func postJSON(url string, payload, dst any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(body)) //nolint:noctx
	if err != nil {
		return fmt.Errorf("POST %s: %w", url, err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, raw)
	}
	return json.Unmarshal(raw, dst)
}

// getJSON GETs url and decodes the response into dst.
func getJSON(url string, dst any) error {
	resp, err := http.Get(url) //nolint:noctx
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, raw)
	}
	return json.Unmarshal(raw, dst)
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

// submitExtraction POSTs to /extract and returns the submission response.
func submitExtraction(req ExtractionRequest) (SubmitResponse, error) {
	var resp SubmitResponse
	err := postJSON(apiBase()+"/extract", req, &resp)
	return resp, err
}

// submitBatch POSTs to /extract/batch and returns the submission response.
func submitBatch(req BatchRequest) (BatchSubmitResponse, error) {
	var resp BatchSubmitResponse
	err := postJSON(apiBase()+"/extract/batch", req, &resp)
	return resp, err
}

// pollTask polls GET /tasks/{id} until state is SUCCESS or FAILURE.
func pollTask(taskID string) (TaskResponse, error) {
	deadline := time.Now().Add(pollTimeoutSec * time.Second)
	for time.Now().Before(deadline) {
		var data TaskResponse
		if err := getJSON(apiBase()+"/tasks/"+taskID, &data); err != nil {
			return data, err
		}
		fmt.Printf("  [%s…] state=%s\n", taskID[:8], data.State)
		if data.State == "SUCCESS" || data.State == "FAILURE" {
			return data, nil
		}
		time.Sleep(pollIntervalSec * time.Second)
	}
	return TaskResponse{}, errors.New("task did not finish within timeout")
}

// ---------------------------------------------------------------------------
// Examples
// ---------------------------------------------------------------------------

func exampleRawText() error {
	fmt.Println("\n── Raw text extraction ──────────────────────────")
	submit, err := submitExtraction(ExtractionRequest{
		RawText: "AGREEMENT dated January 15, 2025 between Acme Corporation " +
			"(Seller) and Beta LLC (Buyer). Purchase price: $12,500 for 500 " +
			"widgets at $25 each. Payment: net 30 days. Governed by Delaware law.",
		Provider:       defaultProvider,
		Passes:         1,
		IdempotencyKey: "demo-raw-text-001",
		ExtractionConfig: ExtractionConfig{
			Temperature: 0.2,
		},
	})
	if err != nil {
		return fmt.Errorf("submit: %w", err)
	}
	fmt.Printf("Submitted: task_id=%s\n", submit.TaskID)

	final, err := pollTask(submit.TaskID)
	if err != nil {
		return err
	}
	entities := []Entity{}
	if final.Result != nil {
		entities = final.Result.Entities
	}
	fmt.Printf("Done — %d entities extracted:\n", len(entities))
	for _, ent := range entities {
		fmt.Printf("  [%s] %q\n", ent.ExtractionClass, ent.ExtractionText)
	}
	return nil
}

func exampleURL() error {
	fmt.Println("\n── URL extraction ───────────────────────────────")
	submit, err := submitExtraction(ExtractionRequest{
		DocumentURL: "https://storage.example.com/contracts/agreement-2025.txt",
		Provider:    defaultProvider,
		ExtractionConfig: ExtractionConfig{
			PromptDescription: "Extract any organisations, dates, and legal terms.",
			Temperature:       0.1,
		},
	})
	if err != nil {
		return fmt.Errorf("submit: %w", err)
	}
	fmt.Printf("Submitted: task_id=%s\n", submit.TaskID)

	final, err := pollTask(submit.TaskID)
	if err != nil {
		return err
	}
	count := 0
	if final.Result != nil {
		count = len(final.Result.Entities)
	}
	fmt.Printf("Done — %d entities extracted.\n", count)
	return nil
}

func exampleBatch() error {
	fmt.Println("\n── Batch extraction ─────────────────────────────")
	submit, err := submitBatch(BatchRequest{
		BatchID: "demo-batch-001",
		Documents: []ExtractionRequest{
			{
				RawText: "Contract A: Acme Corp sells 500 units to Beta LLC for " +
					"$12,500. Delivery Q2 2025.",
			},
			{
				RawText: "Contract B: Charlie Enterprises leases warehouse space " +
					"from Delta Holdings at $3,200/month for 24 months.",
			},
			{
				RawText: "Contract C: Echo Inc purchases software licences from " +
					"Foxtrot SaaS Ltd at $9,000/year, auto-renewing annually.",
			},
		},
		Provider: defaultProvider,
	})
	if err != nil {
		return fmt.Errorf("submit batch: %w", err)
	}

	fmt.Printf("Batch submitted — %d task(s):\n", len(submit.TaskIDs))
	for _, tid := range submit.TaskIDs {
		fmt.Printf("  task_id=%s\n", tid)
	}

	for _, tid := range submit.TaskIDs {
		final, err := pollTask(tid)
		if err != nil {
			return err
		}
		count := 0
		if final.Result != nil {
			count = len(final.Result.Entities)
		}
		fmt.Printf("  [%s…] finished — %d entities\n", tid[:8], count)
	}
	return nil
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	steps := []struct {
		name string
		fn   func() error
	}{
		{"raw text", exampleRawText},
		{"URL", exampleURL},
		{"batch", exampleBatch},
	}

	for _, s := range steps {
		if err := s.fn(); err != nil {
			fmt.Fprintf(os.Stderr, "example %s failed: %v\n", s.name, err)
			os.Exit(1)
		}
	}
}
