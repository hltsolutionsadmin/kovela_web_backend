using DeepFace.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Text.Json;

namespace DeepFace.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class FaceController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private static readonly HttpClient _http = new HttpClient
        {
            BaseAddress = new Uri("http://127.0.0.1:5001/") // Flask server base URL
        };

        private static readonly JsonSerializerOptions JsonOpts = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public FaceController(ApplicationDbContext context)
        {
            _context = context;
        }

        // -------------------------------
        // DTOs (Python service contracts)
        // -------------------------------
        public class B64Request
        {
            public string Base64Image { get; set; } = default!;
        }

        public class EnrollRequestPython
        {
            public string Base64Image { get; set; } = default!;
            public string ExternalId { get; set; } = default!;
        }

        public class EnrollResponsePython
        {
            public string? Error { get; set; }
            public string? Message { get; set; }
            public string? ExternalId { get; set; }
            public float[]? Embedding { get; set; }
            public string? Thumbnail { get; set; }
            public int? Count { get; set; }
        }

        public class CheckRequestPython
        {
            public string Base64Image { get; set; } = default!;
            public int TopK { get; set; } = 5; // now only ask top 5 from Python
            public double Threshold { get; set; } = 0.35;
            public bool IncludeThumbnails { get; set; } = true;
        }

        public class MatchItem
        {
            public string ExternalId { get; set; } = default!;
            public double Score { get; set; }
            public string? Thumbnail { get; set; }
            // Augmented fields from SQL:
            public int? FaceId { get; set; }
        }

        public class CheckResponsePython
        {
            public string? Error { get; set; }
            public string? Type { get; set; }     // "existing" or "new"
            public double BestScore { get; set; }
            public double Threshold { get; set; }
            public List<MatchItem>? Matches { get; set; }
        }

        [HttpPost("check")]
        public async Task<IActionResult> CheckFace([FromBody] B64Request request)
        {
            if (string.IsNullOrWhiteSpace(request.Base64Image))
                return BadRequest("Base64Image is required.");
            if (request.Base64Image.Length < 1000)
                return BadRequest("Base64Image is too short, likely invalid.");

            try
            {
                var pyReq = new CheckRequestPython
                {
                    Base64Image = request.Base64Image,
                    TopK = 5,
                    Threshold = 0.35,
                    IncludeThumbnails = true
                };

                using var resp = await _http.PostAsJsonAsync("check", pyReq, JsonOpts);
                var py = await resp.Content.ReadFromJsonAsync<CheckResponsePython>(JsonOpts);

                if (py == null)
                    return StatusCode(502, new { error = "Python service returned no data" });
                if (!string.IsNullOrEmpty(py.Error))
                    return BadRequest(new { error = py.Error });

                // Filter matches by score (keep only strong matches)
                const double minScore = 0.6; // adjust threshold as needed
                if (py.Matches != null && py.Matches.Any())
                {
                    py.Matches = py.Matches
                        .Where(m => m.Score >= minScore)
                        .OrderByDescending(m => m.Score)
                        .Take(5)
                        .ToList();

                    // Guarantee at least one match if type = existing
                    if (py.Type == "existing" && !py.Matches.Any())
                    {
                        // Re-fetch top candidate (bestScore) from Python
                        var pyReqTop1 = new CheckRequestPython
                        {
                            Base64Image = request.Base64Image,
                            TopK = 1,
                            Threshold = py.Threshold,
                            IncludeThumbnails = true
                        };
                        using var resp2 = await _http.PostAsJsonAsync("check", pyReqTop1, JsonOpts);
                        var pyTop1 = await resp2.Content.ReadFromJsonAsync<CheckResponsePython>(JsonOpts);

                        if (pyTop1?.Matches != null && pyTop1.Matches.Any())
                        {
                            py.Matches = new List<MatchItem> { pyTop1.Matches.First() };
                        }
                    }

                    // Augment with FaceId and fallback thumbnail from SQL
                    if (py.Matches.Any())
                    {
                        var extIds = py.Matches.Select(m => m.ExternalId).Distinct().ToList();
                        var map = _context.Faces
                            .Where(f => extIds.Contains(f.ExternalId))
                            .ToDictionary(f => f.ExternalId, f => new { f.FaceId, f.ThumbnailBase64 });

                        foreach (var m in py.Matches)
                        {
                            if (map.TryGetValue(m.ExternalId, out var faceData))
                            {
                                m.FaceId = faceData.FaceId;
                                // Fallback to database thumbnail if Python service returned empty
                                if (string.IsNullOrEmpty(m.Thumbnail))
                                {
                                    m.Thumbnail = faceData.ThumbnailBase64;
                                }
                            }
                        }

                        // Ensure at least one thumbnail for type == "existing"
                        if (py.Type == "existing" && py.Matches.All(m => string.IsNullOrEmpty(m.Thumbnail)))
                        {
                            // Log warning for debugging
                            Console.WriteLine($"Warning: No thumbnails returned from Python for externalIds: {string.Join(", ", extIds)}");
                            // Try to fetch from database for the top match
                            var topMatch = py.Matches.OrderByDescending(m => m.Score).FirstOrDefault();
                            if (topMatch != null && map.TryGetValue(topMatch.ExternalId, out var topFaceData))
                            {
                                topMatch.Thumbnail = topFaceData.ThumbnailBase64;
                            }
                        }
                    }
                }

                return Ok(py);
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Face check failed: {ex.Message}" });
            }
        }

        [HttpPost("enroll")]
        public async Task<IActionResult> EnrollFace([FromBody] B64Request request)
        {
            if (string.IsNullOrWhiteSpace(request.Base64Image))
                return BadRequest("Base64Image is required.");
            if (request.Base64Image.Length < 1000)
                return BadRequest("Base64Image is too short, likely invalid.");

            try
            {
                // Generate a stable external id (we store this in SQL and pass to Python)
                string externalId = Guid.NewGuid().ToString();

                var pyReq = new EnrollRequestPython
                {
                    Base64Image = request.Base64Image,
                    ExternalId = externalId
                };

                using var resp = await _http.PostAsJsonAsync("enroll", pyReq, JsonOpts);
                var py = await resp.Content.ReadFromJsonAsync<EnrollResponsePython>(JsonOpts);

                if (py == null)
                    return StatusCode(502, new { error = "Python service returned no data" });
                if (!string.IsNullOrEmpty(py.Error))
                    return BadRequest(new { error = py.Error });

                // Save to SQL
                var entity = new FaceEntity
                {
                    ExternalId = externalId,
                    DescriptorJson = JsonSerializer.Serialize(py.Embedding ?? Array.Empty<float>()),
                    ThumbnailBase64 = py.Thumbnail
                };

                _context.Faces.Add(entity);
                await _context.SaveChangesAsync();

                return Ok(new
                {
                    message = "User enrolled successfully",
                    faceId = entity.FaceId,
                    externalId = externalId
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Face enrollment failed: {ex.Message}" });
            }
        }

        [HttpPost("clear")]
        public async Task<IActionResult> ClearAllFaces()
        {
            try
            {
                // 1. Clear database
                _context.Faces.RemoveRange(_context.Faces);
                await _context.SaveChangesAsync();

                // 2. Delete face_db folder
                string dbPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "face_db");
                if (Directory.Exists(dbPath))
                {
                    Directory.Delete(dbPath, true);
                }

                // 3. Delete FAISS index file and thumbnails
                string indexPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "face_index.faiss");
                if (System.IO.File.Exists(indexPath))
                {
                    System.IO.File.Delete(indexPath);
                }
                string thumbnailsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "thumbnails.json");
                if (System.IO.File.Exists(thumbnailsPath))
                {
                    System.IO.File.Delete(thumbnailsPath);
                }

                return Ok(new { message = "All enrolled faces, FAISS index, and thumbnails cleared successfully." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Failed to clear data: {ex.Message}" });
            }
        }
    }
}
